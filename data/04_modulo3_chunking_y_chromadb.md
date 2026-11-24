# Módulo 3 (parte 2) — Chunking, preprocesamiento y persistencia con ChromaDB

Apuntes de la segunda y tercera unidad del módulo 3. En RAG, la calidad de la respuesta no depende solo del LLM sino, sobre todo, de la calidad de los fragmentos que se recuperan.

## Por qué fragmentar (chunking)

Los modelos de lenguaje y, más todavía, los modelos de embeddings tienen una ventana de contexto limitada. Si se convierte un PDF de 50 páginas en un solo vector, se pierden todos los matices: el resultado es un "promedio semántico" tan diluido que no sirve para encontrar respuestas específicas.

El objetivo del chunking es crear **unidades de información atómicas y semánticamente coherentes**.

## Limpieza previa (preprocessing)

Antes de dividir hay que limpiar. Los documentos reales traen ruido:

- encabezados y pies de página repetidos,
- saltos de línea dobles o triples,
- caracteres raros por una mala extracción de tablas,
- rutas de archivos o metadatos basura.

Si un chunk tiene un pie de página en medio de una frase importante, el embedding captura ese ruido y empeora la similitud. "Basura entra, basura sale".

## Estrategias de chunking

**A. Tamaño fijo (fixed-size).** Corta cada X caracteres o tokens. Es la más simple pero la más peligrosa: puede partir una oración o un concepto por la mitad.

**B. Recursive Character Splitting (estándar de la industria).** Es el enfoque preferido en LangChain (`RecursiveCharacterTextSplitter`). Intenta cortar usando una jerarquía de separadores, en este orden:

1. párrafos (`\n\n`),
2. saltos de línea (`\n`),
3. espacios (` `),
4. caracteres individuales, como último recurso.

Así, en la medida de lo posible, los párrafos y las oraciones quedan enteros. En documentos Markdown se pueden agregar los encabezados (`\n## `) al principio de la jerarquía para cortar primero por secciones.

**C. Overlap (solapamiento).** Al dividir, el final del chunk 1 se repite al inicio del chunk 2. Motivación: si un concepto clave cae justo en el límite entre dos fragmentos, el overlap preserva el contexto en ambos y el concepto se puede recuperar sin importar por dónde se cortó. Una configuración típica es 500 tokens por chunk con 50 tokens de overlap.

## Errores comunes de chunking

- **Chunks demasiado chicos:** generan ruido y pierden el contexto global; el LLM no entiende de qué trata el fragmento.
- **Chunks demasiado grandes:** diluyen la precisión del embedding y llenan el contexto del LLM.
- **Medir en caracteres en vez de tokens:** los modelos cobran y limitan por tokens. Hay que validar los chunks con un tokenizador real como `tiktoken`. En LangChain: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)`.
- **No limpiar el texto** antes de fragmentar.

Un buen chunking equilibra **granularidad** (fragmentos chicos para precisión), **contexto** (fragmentos suficientemente grandes, con overlap, para que se entiendan) y **limpieza** (datos normalizados).

## Por qué ChromaDB local

Los embeddings que solo viven en memoria RAM son volátiles: se pierden al reiniciar. Una aplicación real necesita que el conocimiento sobreviva a un reinicio y que se pueda actualizar sin reindexar todo desde cero.

ChromaDB es una base de datos vectorial de código abierto pensada para ser simple. Su modo **local persistente** es ideal para desarrollo y aplicaciones privadas:

- no requiere levantar contenedores Docker,
- guarda los datos en archivos dentro de una carpeta elegida (por ejemplo `./vectorstore`),
- permite operaciones CRUD sobre el "conocimiento" del agente.

Se inicializa con un cliente persistente (`chromadb.PersistentClient(path=...)`, o en LangChain `Chroma(persist_directory=...)`). La métrica de distancia de la colección se elige al crearla, por ejemplo coseno con `hnsw:space = cosine`.

## Operaciones CRUD en el espacio vectorial

- **Create (add / upsert):** no solo se guarda el texto, sino también su vector. Si el texto cambia, el vector se recalcula. La buena práctica es `upsert`: si el ID existe lo actualiza, si no lo crea.
- **Read — `get()` vs `query()`:** `get()` busca por IDs exactos o por filtros de metadatos, como un SELECT tradicional. `query()` busca por **similitud semántica** a partir de un embedding: es el corazón de RAG.
- **Update:** actualiza metadatos o el documento, lo que dispara una nueva vectorización.
- **Delete:** elimina los vectores que ya no son relevantes para el contexto.

## El rol de los metadatos

Los metadatos son etiquetas que se adjuntan a cada vector, por ejemplo `{"autor": "Juan", "fecha": "2024-05", "source": "pdf_v1"}`. Sirven para:

- **filtrar** resultados antes de la búsqueda vectorial,
- **rastrear el origen** de la información que usa el LLM, es decir, citar las fuentes de la respuesta.

## Errores comunes con ChromaDB

- **IDs no deterministas:** usar IDs aleatorios impide actualizar de forma eficiente. Se recomiendan hashes del contenido o IDs estructurados (por ejemplo, archivo + número de chunk).
- **Desajuste de embeddings:** consultar la base con un modelo de embeddings distinto al que se usó para crearla. Los resultados son ruido aleatorio.
- **Ignorar la persistencia:** no inicializar el cliente con una ruta en disco y perder todo al cerrar el script. Además, conviene verificar si la base ya existe antes de reindexar, para ahorrar tiempo y costo.

## RAG: cómo encaja todo

Un sistema RAG (Retrieval-Augmented Generation) tiene dos fases:

1. **Ingesta:** cargar los documentos, limpiarlos, fragmentarlos, calcular sus embeddings y guardarlos en la base vectorial con sus metadatos.
2. **Consulta:** convertir la pregunta del usuario en un embedding con **el mismo modelo**, recuperar los `top_k` fragmentos más similares y pasárselos al LLM junto con la pregunta.

Buenas prácticas de la consulta:

- Mantener un `top_k` de entre 3 y 5. Pasar 50 fragmentos causa errores de límite de tokens y el efecto **"Lost in the Middle"**: el modelo presta menos atención a la información que queda en el medio de un contexto largo.
- Usar un prompt de **grounding** que obligue al modelo a responder solo con el contexto recuperado y a decir "No lo sé" si la respuesta no está, para evitar alucinaciones.
- Validar la salida con un esquema Pydantic que incluya las fuentes, para que cada respuesta sea trazable.
