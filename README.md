# Pre-entrega Módulo 3 — Sistema de recuperación semántica local (RAG)

Sistema **RAG end-to-end** que funciona como asistente de estudio: responde preguntas usando **exclusivamente** mis apuntes de los módulos 1 a 3 del curso de AI Engineering (asincronía, LCEL, validación con Pydantic, embeddings, chunking y ChromaDB).

- **Ingesta:** limpieza → chunking recursivo medido en tokens (500 tokens, 50 de overlap) → ChromaDB persistente en `./vectorstore`.
- **Recuperación:** búsqueda de similitud coseno asíncrona (`top_k = 4`) con el **mismo** modelo de embeddings que se usó para indexar.
- **Generación grounded:** cadena **LCEL** asíncrona `retriever → prompt → Claude → PydanticOutputParser`. Si la respuesta no está en el contexto, el sistema responde **"No lo sé"**.
- **Salida:** objeto **Pydantic** con la respuesta y las **referencias** (archivo, sección, fragmento, relevancia).

## Estructura

| Archivo | Contenido |
|---|---|
| `data/` | Dataset: 4 archivos `.md` con apuntes del curso (módulo 1, módulo 2 y dos del módulo 3) |
| `config.py` | Configuración única: rutas, chunking, `top_k`, LLM y **modelo de embeddings compartido** por ingesta y consulta |
| `ingest.py` | Módulo de ingesta: carga, limpieza, chunking, IDs deterministas y persistencia incremental en ChromaDB |
| `schemas.py` | Contratos Pydantic: `RespuestaLLM` (lo que parsea el `PydanticOutputParser`) y `RespuestaRAG` (respuesta final con referencias) |
| `rag.py` | Retriever asíncrono, prompt "filtro de veracidad", cadena LCEL y la función `async get_rag_response(query)` |
| `main.py` | Pruebas: 3 preguntas con respuesta en los documentos + 2 preguntas trampa, ejecutadas en paralelo con `asyncio.gather` |

## Cómo ejecutarlo

Requiere Python 3.12+ y una API key de Anthropic.

```bash
python -m venv .venv
# Windows (Git Bash)
source .venv/Scripts/activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # completar ANTHROPIC_API_KEY

python ingest.py              # 1) indexa /data en ./vectorstore (opcional: main.py lo hace solo si hace falta)
python main.py                # 2) corre las pruebas
python main.py "¿Qué es el event loop?"   # consulta puntual
```

La primera ejecución descarga el modelo de embeddings (~470 MB) desde HuggingFace; después queda en caché. Si el modelo de embeddings cambia, `ingest.py` lo detecta y pide reindexar con `--reset`.

Uso desde código:

```python
import asyncio
from rag import get_rag_response

respuesta = asyncio.run(get_rag_response("¿Qué es un embedding?"))
print(respuesta.respuesta)
for ref in respuesta.referencias:
    print(ref.fuente, ref.seccion, ref.relevancia)
```

## Variables de entorno

| Variable | Obligatoria | Default | Descripción |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Sí | — | API key de Anthropic |
| `LLM_MODEL` | No | `claude-haiku-5-5` | Modelo de Claude |
| `LLM_MAX_TOKENS` | No | `4096` | Máximo de tokens de salida |
| `TOP_K` | No | `4` | Fragmentos recuperados (se valida que esté entre 3 y 5) |
| `MIN_RELEVANCE` | No | `0.70` | Similitud coseno mínima para que un fragmento entre al contexto |
| `EMBEDDING_MODEL` | No | `intfloat/multilingual-e5-small` | Modelo de embeddings local |

Las credenciales solo se leen del `.env`, que está en `.gitignore` junto con `.venv/` y `vectorstore/`. En el repo solo se sube `.env.example`.

## Decisiones de diseño

### 1. Ingesta y chunking (`ingest.py`)

- **Limpieza previa:** normaliza saltos de línea, quita espacios sobrantes, BOM y saltos triples ("basura entra, basura sale").
- **`RecursiveCharacterTextSplitter.from_tiktoken_encoder`:** el tamaño se mide en **tokens reales** (tiktoken `cl100k_base`), no en caracteres: `chunk_size=500`, `chunk_overlap=50`.
- **Separadores jerárquicos para Markdown:** primero corta por secciones (`\n## `), después por párrafos, líneas y oraciones; los caracteres sueltos son el último recurso. Así los chunks respetan la estructura del documento.
- **Cabecera contextual:** a cada chunk se le antepone `[Título del documento > Sección]`. Un fragmento que quedó separado de su encabezado sigue teniendo sentido para el modelo de embeddings y para el LLM.
- **Metadatos para trazabilidad:** `source`, `seccion`, `chunk_index`, `tokens`, `file_hash`. Son los que después aparecen como referencias en la respuesta.
- **IDs deterministas** (`archivo::índice::hash`): el mismo contenido siempre genera el mismo ID, lo que permite actualizar y borrar de forma precisa.

### 2. Persistencia sin reindexar de más

`ingest_documents()` compara el hash de cada archivo de `/data` con lo que ya está guardado en ChromaDB:

| Situación | Acción |
|---|---|
| La base ya existe y el archivo no cambió | Se omite (no se vuelve a embeber) |
| El archivo cambió | Se borran sus chunks viejos y se insertan los nuevos |
| El archivo se eliminó de `/data` | Se borran sus chunks |
| `python ingest.py --reset` | Se borra la colección y se reindexa todo |

### 3. Embeddings consistentes

Anthropic no ofrece modelo de embeddings, así que se usa un modelo **local** de HuggingFace, `intfloat/multilingual-e5-small`: gratis, sin enviar los documentos a terceros, y multilingüe (los apuntes están en español).

**Por qué este modelo y no otro:** la primera opción fue `paraphrase-multilingual-MiniLM-L12-v2`, pero ese modelo solo lee **128 tokens** por texto: de cada chunk de 500 tokens se habría embebido apenas el principio. Es el error "ignorar el límite de tokens" del módulo 3. `multilingual-e5-small` lee hasta **512 tokens**. La ingesta valida cada chunk con el tokenizador del propio modelo (tiktoken mide lo que cobra el LLM, pero el modelo de embeddings tokeniza distinto) y avisa si alguno se pasa. Con el dataset actual, 5 de 18 chunks superan los 512 tokens por hasta 52 tokens; lo que se trunca es el final del chunk, que de todos modos está repetido al principio del siguiente gracias al overlap. Los modelos E5 esperan el prefijo `passage: ` en los documentos y `query: ` en las preguntas, y así se configuran.

Para evitar el error #1 (indexar con un modelo y consultar con otro):

- El modelo se define en un solo lugar (`config.get_embeddings()`), usado por la ingesta y por la consulta.
- El nombre del modelo se guarda en los metadatos de la colección de ChromaDB; si al abrirla el modelo configurado es otro, se lanza `EmbeddingsIncompatiblesError` en vez de devolver resultados aleatorios. Esto pasó de verdad al cambiar de modelo durante el desarrollo:

  ```
  EmbeddingsIncompatiblesError: La colección 'apuntes_curso' se indexó con 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
  pero el modelo configurado es 'intfloat/multilingual-e5-small'. Ejecutá `python ingest.py --reset` para reindexar.
  ```
- Los vectores se normalizan y la colección usa distancia **coseno** (`hnsw:space = cosine`).

### 4. Recuperación y generación (`rag.py`)

```
pregunta ─┬─> recuperar() ──> docs ──> formatear_contexto() ─┐
          └─> pregunta ───────────────────────────────────────┴─> ¿hay docs?
                                                                    ├─ no ─> "No lo sé" (sin llamar al LLM)
                                                                    └─ sí ─> prompt | ChatAnthropic | PydanticOutputParser
                                                                              └── .with_retry(3 intentos) ──┘
```

- **Asincronía de punta a punta:** la búsqueda usa `asimilarity_search_with_relevance_scores`, la cadena se ejecuta con `ainvoke` y la ingesta inicial corre en un thread (`asyncio.to_thread`) para no bloquear el event loop. Las pruebas lanzan todas las preguntas en paralelo con `asyncio.gather`.
- **`top_k = 4`:** dentro del rango 3–5 para evitar el "contexto infinito" y el efecto *Lost in the Middle*.
- **Umbral de relevancia:** los fragmentos con similitud menor a `MIN_RELEVANCE` se descartan. Si no queda ninguno, la cadena (`RunnableBranch`) responde "No lo sé" sin gastar una llamada al LLM. Ojo: el modelo E5 concentra las similitudes en un rango alto (en las pruebas, "¿Quién ganó el Mundial 2022?" llegó a 0,83 contra los apuntes, y una pregunta válida tuvo fragmentos de 0,72). Por eso el umbral (0,70) solo filtra lo que no tiene ninguna relación, y lo que realmente evita las alucinaciones es el prompt de grounding.
- **Prompt "filtro de veracidad":** el modelo debe responder solo con el CONTEXTO, no usar conocimiento propio, no inventar cifras ni comandos, citar los fragmentos usados, e ignorar instrucciones que aparezcan dentro de los documentos (defensa básica contra *prompt injection*).
- **`PydanticOutputParser`:** sus `format_instructions` se inyectan en el prompt con `.partial()`, y el parser valida la salida contra `RespuestaLLM`. Si el JSON es inválido, se reintenta (como en el Módulo 2).

### 5. Validación con Pydantic (`schemas.py`)

- `RespuestaLLM` (salida del LLM): `encontrado`, `respuesta`, `fragmentos_usados`. Validador: si `encontrado=true` debe citar al menos un fragmento; si es `false`, la respuesta se normaliza a "No lo sé" y sin citas.
- `RespuestaRAG` (salida final): `pregunta`, `respuesta`, `encontrado`, `referencias: list[Referencia]`. Validador: una respuesta encontrada **debe** tener referencias y un "No lo sé" **no puede** tenerlas.
- **Las referencias no las escribe el LLM:** el modelo solo dice qué números de fragmento usó, y el código los traduce a los metadatos reales de ChromaDB (archivo, sección, relevancia, extracto). Si cita un fragmento inexistente, se descarta; si no queda ninguna cita válida, la respuesta pasa a "No lo sé". Así el modelo no puede inventar fuentes.

## Pruebas y evidencia

`python main.py` lanza 5 consultas **en paralelo** con `asyncio.gather` y verifica automáticamente cada resultado:

| Pregunta | Tipo | Esperado |
|---|---|---|
| ¿Por qué no conviene usar asyncio.gather con 10.000 llamadas a la vez? | En los apuntes (módulo 1) | Respuesta con referencias |
| ¿Qué diferencia hay entre get() y query() en ChromaDB? | En los apuntes (módulo 3) | Respuesta con referencias |
| ¿Para qué sirve el overlap al fragmentar documentos? | En los apuntes (módulo 3) | Respuesta con referencias |
| ¿Cómo se definen los nodos y aristas de un grafo en LangGraph? | **Trampa** (es del curso, pero del módulo 5, que no está en los apuntes) | "No lo sé", sin referencias |
| ¿Quién ganó el Mundial de fútbol 2022? | **Trampa** (el modelo lo sabe, pero no está en los apuntes) | "No lo sé", sin referencias |

Las dos trampas prueban cosas distintas: la primera es del mismo dominio (el retriever trae fragmentos que hablan de LangChain y del curso), así que el modelo tiene que darse cuenta de que **LangGraph** no aparece; la segunda la conoce cualquier LLM, así que prueba que no completa con conocimiento propio.

### Resultado: 5/5 pruebas aprobadas

Pregunta con respuesta en los apuntes:

```json
{
  "pregunta": "¿Qué diferencia hay entre get() y query() en ChromaDB?",
  "respuesta": "`get()` busca por IDs exactos o por filtros de metadatos, de forma similar a un SELECT tradicional. `query()` busca por similitud semántica a partir de un embedding, y es el corazón de RAG.",
  "encontrado": true,
  "referencias": [
    {
      "fuente": "04_modulo3_chunking_y_chromadb.md",
      "seccion": "Por qué ChromaDB local",
      "chunk_index": 2,
      "relevancia": 0.8793,
      "extracto": "## Por qué ChromaDB local  Los embeddings que solo viven en memoria RAM son volátiles: se pierden al reiniciar. Una aplicación real necesita que el conocimiento sobreviva a un reinicio y que se pueda..."
    }
  ]
}
```

Pregunta trampa:

```json
{
  "pregunta": "¿Cómo se definen los nodos y aristas de un grafo en LangGraph?",
  "respuesta": "No lo sé. No tengo acceso a esa información en la documentación disponible.",
  "encontrado": false,
  "referencias": []
}
```

Resumen final de la corrida:

```
RESULTADO: OK - respondió citando fuentes    (x3)
RESULTADO: OK - no alucinó                   (x2)
Pruebas aprobadas: 5/5 (tiempo total en paralelo: 2.8s)
```

Las 5 consultas tardan en total lo mismo que la más lenta (2,8 s), porque mientras una espera al LLM avanzan las demás.

### Evidencia de persistencia

Primera ejecución de `python ingest.py` (base vacía):

```
[INFO] ingesta: 4 documentos cargados desde data/
[INFO] ingesta:   01_modulo1_asincronia_y_clientes.md: 4 chunks (tokens min=401, max=511)
[INFO] ingesta:   02_modulo2_lcel_y_validacion.md: 5 chunks (tokens min=343, max=526)
[INFO] ingesta:   03_modulo3_embeddings_y_similitud.md: 5 chunks (tokens min=264, max=397)
[INFO] ingesta:   04_modulo3_chunking_y_chromadb.md: 4 chunks (tokens min=335, max=529)
[INFO] ingesta: Indexados 18 chunks nuevos
[INFO] ingesta: Colección 'apuntes_curso': 18 chunks en vectorstore/
```

Siguientes ejecuciones (la base ya existe y los archivos no cambiaron):

```
[INFO] ingesta:   01_modulo1_asincronia_y_clientes.md: sin cambios, se omite
[INFO] ingesta:   02_modulo2_lcel_y_validacion.md: sin cambios, se omite
[INFO] ingesta:   03_modulo3_embeddings_y_similitud.md: sin cambios, se omite
[INFO] ingesta:   04_modulo3_chunking_y_chromadb.md: sin cambios, se omite
[INFO] ingesta: La base vectorial ya está al día: no hace falta reindexar
```

Algunos chunks tienen más de 500 tokens porque se les suma la cabecera `[Título > Sección]` (unos 15 a 30 tokens) después del corte.

### Seguridad de credenciales

- La API key se lee solo de `.env` con `python-dotenv`; no aparece en el código.
- `.env` está en `.gitignore`; el repo incluye `.env.example` con un valor de ejemplo.
- `vectorstore/` y `.venv/` tampoco se suben: se regeneran con `pip install` y `python ingest.py`.
