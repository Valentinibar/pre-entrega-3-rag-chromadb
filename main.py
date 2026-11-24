"""Pruebas del sistema RAG: preguntas con respuesta en los documentos y preguntas trampa.

Uso:
    python main.py                          # corre la batería de pruebas
    python main.py "¿tu pregunta aquí?"     # hace una consulta puntual
"""

import asyncio
import sys
import time

from rag import get_rag_response, get_vectorstore

# (pregunta, ¿la respuesta está en los documentos?)
PRUEBAS = [
    ("¿Por qué no conviene usar asyncio.gather con 10.000 llamadas a la vez?", True),
    ("¿Qué diferencia hay entre get() y query() en ChromaDB?", True),
    ("¿Para qué sirve el overlap al fragmentar documentos?", True),
    # Preguntas trampa: la respuesta NO está en los apuntes
    ("¿Cómo se definen los nodos y aristas de un grafo en LangGraph?", False),
    ("¿Quién ganó el Mundial de fútbol 2022?", False),
]


def imprimir(respuesta, segundos: float) -> None:
    print(respuesta.model_dump_json(indent=2))
    print(f"(tiempo: {segundos:.1f}s)")


async def consultar(pregunta: str):
    inicio = time.perf_counter()
    respuesta = await get_rag_response(pregunta)
    return respuesta, time.perf_counter() - inicio


async def correr_pruebas() -> bool:
    await get_vectorstore()  # inicializa la base (e indexa solo si hace falta) antes del gather

    # Todas las consultas se lanzan en paralelo: mientras una espera al LLM, avanzan las otras
    inicio = time.perf_counter()
    resultados = await asyncio.gather(
        *(consultar(p) for p, _ in PRUEBAS), return_exceptions=True
    )
    total = time.perf_counter() - inicio

    aprobadas = 0
    for (pregunta, esperado), resultado in zip(PRUEBAS, resultados):
        tipo = "EN LOS DOCUMENTOS" if esperado else "PREGUNTA TRAMPA"
        print(f"\n{'=' * 70}\n[{tipo}] {pregunta}\n{'=' * 70}")

        if isinstance(resultado, Exception):
            print(f"ERROR: {type(resultado).__name__}: {resultado}")
            continue

        respuesta, segundos = resultado
        imprimir(respuesta, segundos)

        ok = respuesta.encontrado == esperado
        if esperado:
            ok = ok and len(respuesta.referencias) > 0
        else:
            ok = ok and respuesta.respuesta.startswith("No lo sé") and not respuesta.referencias
        aprobadas += ok
        print("RESULTADO:", "OK" if ok else "FALLÓ",
              "- respondió citando fuentes" if esperado else "- no alucinó")

    print(f"\n{'=' * 70}\nPruebas aprobadas: {aprobadas}/{len(PRUEBAS)} "
          f"(tiempo total en paralelo: {total:.1f}s)")
    return aprobadas == len(PRUEBAS)


async def main() -> None:
    if len(sys.argv) > 1:
        respuesta, segundos = await consultar(" ".join(sys.argv[1:]))
        imprimir(respuesta, segundos)
        return
    if not await correr_pruebas():
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
