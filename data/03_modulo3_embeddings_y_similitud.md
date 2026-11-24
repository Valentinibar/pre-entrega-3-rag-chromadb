# Módulo 3 (parte 1) — Embeddings y métricas de similitud

Apuntes de la primera unidad del módulo 3, "Persistencia de datos y vector DBs". Objetivo del módulo: implementar memoria de largo plazo con arquitecturas RAG básicas y bases de datos vectoriales locales.

## El problema de la búsqueda por palabras clave

Si se busca "cómo limitar el tráfico de red" en un buscador por palabras clave, y el documento correcto habla de "restricción de ancho de banda" o "rate-limiting", probablemente no aparezca. Ese es el talón de Aquiles de la búsqueda tradicional: la **rigidez terminológica**.

La IA lo resuelve transformando el lenguaje en geometría: los textos se convierten en vectores y se mide qué tan cerca están dos ideas en un espacio de muchas dimensiones. Así la búsqueda entiende el **significado** (semántica) y no solo las letras exactas.

## Qué es un embedding

Un embedding es una **representación numérica densa** de un objeto (normalmente texto, pero también imágenes o audio) en un espacio vectorial de alta dimensionalidad. Dicho simple: una lista de números que captura el significado de un fragmento de texto.

**De lo discreto a lo denso.** Antes se usaba *one-hot encoding*: cada palabra era una posición en un vector gigante lleno de ceros (con un vocabulario de 50.000 palabras, "perro" era un vector de 50.000 elementos con un solo 1). El problema: la distancia entre "perro" y "canino" era igual a la distancia entre "perro" y "microprocesador". No había noción de significado.

Los embeddings modernos son **densos**: en los modelos de OpenAI suelen tener entre 1536 y 3072 dimensiones, y cada dimensión representa una característica abstracta aprendida durante el entrenamiento. Por ejemplo, `text-embedding-3-small` de OpenAI devuelve 1536 números flotantes por texto.

**Analogía del mapa.** En un mapa 2D, la distancia entre Madrid y Lisboa refleja su cercanía geográfica. Un embedding hace lo mismo en miles de dimensiones: ubica "rey" cerca de "hombre" y "reina" cerca de "mujer", conservando relaciones lógicas entre conceptos.

## Métricas de similitud

**Similitud coseno (la más usada).** Mide el coseno del ángulo entre dos vectores, es decir, captura la **dirección** e ignora la longitud: no importa si un texto es más largo, sino que "apunten" al mismo concepto. Va de -1 a 1; en embeddings de texto suele moverse entre 0 y 1, donde 1 es identidad total. Es el estándar en RAG porque se enfoca en el contenido temático.

**Distancia euclídea (L2).** Mide la distancia en línea recta entre dos puntos. Es muy sensible a la magnitud: un párrafo y una oración sobre el mismo tema pueden quedar lejos. Se usa sobre todo con vectores normalizados.

**Producto punto (dot product).** Multiplica las magnitudes por el coseno del ángulo. Común en sistemas de recomendación, donde la magnitud (la "importancia") del vector es relevante.

Regla de oro: las bases vectoriales permiten elegir la métrica; con modelos de OpenAI o HuggingFace, la similitud coseno es casi siempre la elección correcta si los vectores no están normalizados. Si se normalizan, coseno y producto punto dan el mismo orden de resultados.

Ejemplo conceptual: "El sistema de archivos está lleno" y "No queda espacio en el disco duro" tendrían una similitud coseno alta (cerca de 0,85), mientras que "Me gusta comer manzanas" quedaría muy lejos (cerca de 0,15).

## Modelos estáticos vs. contextuales

- **Estáticos (Word2Vec, GloVe):** cada palabra tiene un único vector. "Banco" (para sentarse) y "banco" (entidad financiera) tienen el mismo embedding: el modelo es ciego al contexto.
- **Contextuales (BERT, Ada, Voyage):** la misma palabra genera vectores distintos según las palabras que la rodean. En "me senté en el banco" el modelo entiende que se trata de mobiliario urbano.

Como desarrolladores usamos estos modelos a través de APIs (OpenAI, Voyage, Cohere) o localmente con librerías como `sentence-transformers` de HuggingFace, que no tienen costo por llamada y no envían los datos a terceros.

## Aplicaciones reales

- **Búsqueda semántica:** buscadores como los de Notion o GitHub. Una búsqueda de "problemas de pagos" puede encontrar "error en la pasarela de Stripe" o "transacción declinada" sin compartir palabras.
- **Deduplicación:** en soporte técnico se agrupan tickets que preguntan lo mismo con distintas palabras ("¿cómo cambio mi clave?" y "procedimiento para resetear password").
- **Clasificación zero-shot:** se calcula el embedding de cada etiqueta (URGENTE, SOLICITUD, SPAM) y el del correo entrante; el correo pertenece a la categoría más cercana, sin entrenar ningún modelo.

## Errores comunes con embeddings

- **Creer que entienden todo:** capturan correlaciones estadísticas, no lógica. Pueden confundir la negación: "hay stock" y "no hay stock" quedan cerca porque ambos hablan de disponibilidad.
- **Ignorar el límite de tokens:** cada modelo de embeddings tiene una ventana de contexto. Embeber un PDF de 100 páginas de una vez trunca el texto o diluye el significado. Por eso existe el chunking.
- **Mezclar modelos:** nunca comparar un embedding de OpenAI con uno de un modelo local de HuggingFace. Viven en espacios vectoriales distintos, con dimensiones distintas, y no hay relación matemática entre ellos. Indexar con un modelo y consultar con otro da resultados aleatorios.
- **Domain shift:** un modelo entrenado con texto general de internet puede medir mal la similitud en lenguaje médico muy especializado o en código fuente.

## Síntesis

- Los embeddings traducen el lenguaje a coordenadas numéricas.
- La similitud coseno mide qué tan cerca están dos ideas.
- Son la base de la memoria de largo plazo de los sistemas de IA y el motor de búsqueda dentro de una arquitectura RAG (Retrieval-Augmented Generation).
