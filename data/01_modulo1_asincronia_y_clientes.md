# Módulo 1 — La interfaz base: asincronía en Python 3.12 y clientes de LLM

Apuntes del primer módulo del curso de AI Engineering. Objetivo del módulo: construir la capa base para hablar con modelos de lenguaje usando Python 3.12, asyncio y los SDKs oficiales de OpenAI y Anthropic.

## Por qué asyncio en ingeniería de IA

En una aplicación con LLMs casi todo el tiempo se pasa esperando, no calculando: el cálculo pesado ocurre en los servidores del proveedor, y nuestro programa espera a que el prompt viaje, a que el modelo procese y a que los tokens vuelvan por la red.

Ejemplo de clase: un agente que (1) busca en una base vectorial como ChromaDB, (2) consulta una API externa de clima y (3) genera la respuesta con un LLM. En un modelo síncrono, si cada paso tarda 2 segundos, el usuario espera 6 segundos como mínimo, y mientras el servidor espera no puede atender a nadie más.

Asyncio permite que un solo proceso maneje muchas tareas de entrada/salida (I/O) de forma concurrente. Importante: **no es paralelismo real** (varios procesadores), sino **concurrencia cooperativa**: cuando la tarea A queda esperando datos de la red, cede el control para que avance la tarea B.

Python 3.12 refina la API de asyncio y elimina comportamientos ambiguos heredados (por ejemplo, la creación automática de loops), para forzar patrones más limpios y predecibles.

## Conceptos fundamentales de asyncio

- **Corrutina (`async def` + `await`):** función que puede pausar su ejecución. Analogía: una receta con señaladores; cuando hay que esperar que el horno caliente (`await`), se marca el punto y mientras tanto se avanza con otra receta.
- **Event loop:** el motor que coordina las corrutinas. Analogía: el gestor de un restaurante que, si un mozo está esperando un plato de la cocina, lo manda a atender otra mesa.
- **Task:** una corrutina programada para ejecutarse "lo antes posible" en el event loop. Una corrutina sola no hace nada hasta que se la espera; una Task corre en segundo plano mientras el código sigue con otras cosas.

## Patrones de concurrencia

**Punto de entrada: `asyncio.run(main())`.** Es el estándar para arrancar la aplicación: crea el loop, termina las tareas y cierra los ejecutores de forma limpia. Ya no conviene usar `asyncio.get_event_loop()` a la ligera; si hace falta el loop desde adentro de una función, se usa `asyncio.get_running_loop()`.

**`asyncio.gather`:** el patrón más común para disparar varias llamadas a LLMs al mismo tiempo, por ejemplo para comparar las respuestas de tres modelos sobre el mismo prompt. Conviene pasar `return_exceptions=True` para que el fallo de una llamada no tumbe a todas: el error vuelve como un resultado más. En un pipeline RAG se puede usar `gather` para buscar en la base vectorial y en un buscador web al mismo tiempo.

**`asyncio.TaskGroup`** (desde 3.11, refinado en 3.12): alternativa más segura a `gather` para grupos de tareas, porque si una falla, las demás se cancelan limpiamente.

## Resiliencia: timeouts y semáforos

**Timeouts.** Nunca hay que dejar una llamada a un LLM abierta indefinidamente. Se usa `async with asyncio.timeout(10.0):` alrededor de la llamada y se captura `TimeoutError` para aplicar una estrategia de fallback.

**Semáforos.** Si se disparan 1.000 llamadas en un segundo con `gather`, el proveedor responde `429 Too Many Requests`. Un `asyncio.Semaphore(5)` funciona como un portero: solo deja que N tareas estén activas a la vez dentro de `async with sem:`; el resto espera su turno.

Pregunta de reflexión del módulo: para generar embeddings de 10.000 filas **no** se hace un `gather` de 10.000 llamadas simultáneas (llevaría a errores 429). Se encolan todas las tareas, pero un semáforo garantiza que solo N corran en simultáneo, y cada llamada lleva su propio timeout. En el ejemplo resuelto de clase, con 8 textos y `Semaphore(3)`, se procesan en unas 3 tandas, y los que superan el timeout de 2 segundos se marcan como fallidos sin romper el lote.

## Anti-patrones

- **"Async hace más rápido el procesamiento de datos": falso.** Asyncio no acelera cálculos. Para tareas CPU-bound (limpieza pesada con Pandas, inferencia local con PyTorch) hace falta `multiprocessing`. Asyncio sirve para tareas I/O-bound: red, bases de datos, APIs.
- **Bloquear el loop.** Llamar a una función síncrona lenta dentro de una corrutina (por ejemplo `time.sleep(5)` en lugar de `await asyncio.sleep(5)`) congela a todos los usuarios. Si hay que usar una librería que no es asíncrona, se la envuelve con `await asyncio.to_thread(funcion_bloqueante, datos)`, que la ejecuta en otro hilo.

## Streaming de tokens

Los usuarios esperan ver las palabras aparecer de a una. Esto se logra con **generadores asíncronos**: en vez de esperar los 500 tokens de la respuesta completa, se procesa cada token apenas llega y se envía al frontend por WebSockets o Server-Sent Events (SSE). Así baja el **Time to First Token (TTFT)**, una de las métricas de experiencia de usuario más importantes en productos de IA.

## Abstracción de clientes y configuración robusta

Instanciar el cliente de OpenAI o Anthropic directamente dentro de la lógica de negocio trae tres problemas:

1. **Acoplamiento:** cambiar de proveedor obliga a reescribir muchas líneas.
2. **Testing difícil:** es complicado simular (mockear) la API si el cliente está enterrado en la lógica.
3. **Inconsistencia:** cada SDK responde distinto. OpenAI devuelve el texto en `choices[0].message`; Anthropic en `content`, que es una lista de bloques.

La solución vista es el **patrón Factory**: una clase base abstracta común y un gestor que decide en tiempo de ejecución qué cliente instanciar según la configuración (por ejemplo, la variable `LLM_PROVIDER`).

La configuración se valida con **Pydantic**: rangos de parámetros como la temperatura o `max_tokens`, y las API keys guardadas como `SecretStr`, para que no aparezcan por accidente en logs ni mensajes de error. Las llaves se leen siempre de variables de entorno (archivo `.env`), nunca se escriben en el código.

## Síntesis del módulo 1

- Usar `asyncio.run()` como único punto de entrada.
- No bloquear nunca el loop; envolver código síncrono con `to_thread()`.
- Timeouts y semáforos desde el primer día: un sistema de IA sin control de flujo es una bomba de tiempo de cuotas agotadas y latencias extremas.
- Abstraer los proveedores detrás de una interfaz común y validar la configuración con Pydantic.
