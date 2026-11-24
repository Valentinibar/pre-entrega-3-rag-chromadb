# Módulo 2 — Encadenamiento lógico: LCEL, runnables y validación estructurada

Apuntes del segundo módulo. Objetivo: pasar de scripts sueltos que llaman a un modelo a pipelines declarativos, asíncronos y robustos con LangChain.

## Por qué LCEL

Las aplicaciones reales rara vez son una sola llamada al modelo. Un análisis de sentimiento puede requerir traducir, clasificar, extraer entidades en JSON y resumir. Hacerlo con código imperativo (muchos `if`, `await` y transformaciones manuales) termina en código espagueti difícil de testear.

Con los SDKs base, cada paso es manual: formatear variables dentro de un string, llamar al modelo manejando reintentos, y parsear la respuesta. Agregar streaming, paralelismo o trazabilidad multiplica la complejidad.

**LCEL (LangChain Expression Language)** es un lenguaje declarativo para componer cadenas donde cada componente es un bloque con interfaz estándar. Lo que se obtiene "gratis":

- **Async de primera clase:** se alterna entre síncrono y asíncrono sin cambiar la lógica.
- **Streaming automático** si el modelo lo soporta.
- **Paralelismo** de tareas independientes sin escribir `asyncio.gather` a mano.
- **Observabilidad** integrada con herramientas como LangSmith.

## El protocolo Runnable

Casi todo en LangChain (modelos, prompts, parsers, herramientas) hereda de `Runnable`, así que todos implementan los mismos métodos:

| Método | Qué hace |
|---|---|
| `invoke()` | Ejecuta con una sola entrada (síncrono) |
| `ainvoke()` | Versión asíncrona de `invoke` |
| `batch()` / `abatch()` | Ejecuta con una lista de entradas, optimizando la latencia |
| `stream()` / `astream()` | Devuelve la respuesta en fragmentos a medida que se genera |

Analogía de clase: una fábrica de autos donde cada estación (Runnable) recibe piezas y entrega un componente más complejo; el operador `|` es la cinta transportadora que las une.

## Composición con el operador pipe

La salida del componente de la izquierda es la entrada del de la derecha. La estructura típica es `PromptTemplate | ChatModel | OutputParser`:

1. **Prompt** (diccionario → PromptValue): recibe las variables y arma el mensaje.
2. **Modelo** (PromptValue → Message): envía el mensaje al LLM y devuelve un `BaseMessage`.
3. **Parser** (Message → string/objeto): limpia la respuesta cruda. `StrOutputParser` extrae el `.content` como string.

## Prompts como artefactos de software

`ChatPromptTemplate.from_messages` permite definir roles (system, human, ai). El rol system es clave para que el modelo siga instrucciones de seguridad o formato. Los prompts deben ser modulares y versionables; las variables se definen entre llaves (`{texto}`) y no con f-strings. Con `.partial()` se fijan de antemano variables que no cambian, como las instrucciones de formato.

**Input shaping:** en vez de preparar los datos antes de la cadena, se modelan adentro con `RunnablePassthrough` o funciones lambda. Por ejemplo, si el prompt espera `context` y `question` pero solo llega la pregunta, un diccionario `{"context": retriever, "question": RunnablePassthrough()}` al inicio de la cadena completa la entrada. LCEL convierte automáticamente cualquier función en un Runnable.

## Paralelismo y lotes

- **`RunnableParallel`:** ejecuta varias sub-cadenas a la vez sobre la misma entrada y devuelve un diccionario con una clave por rama. El tiempo total es el de la rama más lenta, no la suma. Ejemplo de clase: evaluar una idea de negocio desde una perspectiva técnica y otra financiera al mismo tiempo.
- **`abatch`:** para procesar muchas entradas (por ejemplo 100 reseñas) es mucho más eficiente que un `for` con `ainvoke`.

Patrón general de un pipeline de producción: descomponer en sub-cadenas, paralelizar lo independiente y sintetizar al final. Las claves del diccionario que produce una etapa deben coincidir con las variables del prompt de la etapa siguiente.

## Errores comunes con LCEL

- **Mismatch de tipos:** si el prompt tiene variables, la entrada debe ser un diccionario (`{"text": "valor"}`), no un string suelto.
- **Olvidar el parser:** sin parser se recibe un `AIMessage` en lugar de texto.
- **No usar métodos asíncronos en aplicaciones web:** en FastAPI conviene usar siempre `ainvoke`, `abatch` y `astream` para no bloquear el event loop.

## Validación estructurada con Pydantic

Los LLMs son probabilísticos y su salida natural es texto libre, lo cual es "veneno" para la escalabilidad: pueden cambiar el formato, inventar claves JSON o devolver un string donde se esperaba un número. La solución es un **contrato de datos** con Pydantic, que valida en tres niveles:

1. **Sintáctico:** el JSON es íntegro y parseable.
2. **Estructural:** están todos los campos requeridos.
3. **Semántico:** los valores cumplen reglas de negocio (por ejemplo, una confianza entre 0 y 1). Se implementa con `Field(...)` y validadores (`field_validator`, `model_validator`).

Dos formas de obtener salida estructurada en LangChain:

- **`model.with_structured_output(Esquema)`:** usa tool calling o JSON mode del proveedor para que el modelo "encaje" la respuesta en el molde.
- **`PydanticOutputParser(pydantic_object=Esquema)`:** genera instrucciones de formato (`get_format_instructions()`) que se insertan en el prompt, y después parsea y valida el texto que devuelve el modelo. Si no cumple el esquema, lanza `OutputParserException`.

Además hay que revisar el `finish_reason` / `stop_reason`: si el modelo cortó la respuesta por falta de tokens (`max_tokens`), el JSON queda incompleto y hay que reintentar.

## Resiliencia: reintentos y fallbacks

Las llamadas pueden fallar por:

- **Errores transitorios:** caídas de red, latencia, límite de tasa (429).
- **Errores de formato:** el modelo no siguió el esquema.

Estrategias:

- **Reintentos con backoff exponencial:** esperar cada vez un poco más antes de reintentar, para no saturar el servicio. En LCEL: `.with_retry(retry_if_exception_type=..., stop_after_attempt=3, wait_exponential_jitter=True)`.
- **Fallbacks:** si un modelo falla repetidamente, redirigir a un modelo más liviano o a una respuesta predefinida (`.with_fallbacks([...])`).

Errores de arquitectura a evitar: confiar ciegamente en `json.loads` sin validar el esquema, y **reintentar errores permanentes** (como una API key inválida o un prompt con un error lógico), que solo desperdicia dinero y latencia. Hay que distinguir errores de transporte de errores de lógica.

## Síntesis del módulo 2

- Componer con `|`: prompt → modelo → parser.
- Todos los componentes comparten la interfaz Runnable (`invoke`, `batch`, `stream` y sus versiones async).
- El código describe el qué (el flujo), no el cómo (hilos y estados).
- Toda salida de un LLM que alimente a otro sistema debe pasar por un esquema Pydantic, con reintentos solo para errores recuperables.
