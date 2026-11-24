"""Módulo de ingesta: lee /data, limpia, fragmenta por tokens y persiste en ChromaDB.

Uso:
    python ingest.py            # ingesta incremental (solo procesa archivos nuevos o modificados)
    python ingest.py --reset    # borra la colección y reindexa todo desde cero
"""

import argparse
import hashlib
import logging
import re
from pathlib import Path

import chromadb
import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

log = logging.getLogger("ingesta")

_tokenizer = tiktoken.get_encoding(config.TOKENIZER_ENCODING)


class EmbeddingsIncompatiblesError(RuntimeError):
    """La colección fue creada con un modelo de embeddings distinto al configurado."""


# --------------------------------------------------------------------------- #
# 1. Carga y limpieza
# --------------------------------------------------------------------------- #
def contar_tokens(texto: str) -> int:
    return len(_tokenizer.encode(texto))


def hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def limpiar_texto(texto: str) -> str:
    """Preprocesamiento: 'basura entra, basura sale'."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = texto.replace(" ", " ").replace("﻿", "")       # espacios raros / BOM
    texto = re.sub(r"[^\S\n]+\n", "\n", texto)                        # espacios al final de línea
    texto = re.sub(r"[ \t]{2,}", " ", texto)                          # espacios múltiples
    texto = re.sub(r"\n{3,}", "\n\n", texto)                          # saltos de línea triples
    return texto.strip()


def cargar_documentos(data_dir: Path = config.DATA_DIR) -> list[Document]:
    """Lee todos los .txt / .md de la carpeta y devuelve un Document por archivo."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de datos: {data_dir}")

    documentos = []
    for ruta in sorted(data_dir.iterdir()):
        if ruta.suffix.lower() not in config.EXTENSIONES_VALIDAS:
            continue
        texto = limpiar_texto(ruta.read_text(encoding="utf-8"))
        if not texto:
            log.warning(f"Archivo vacío, se ignora: {ruta.name}")
            continue
        titulo = next(
            (l.lstrip("# ").strip() for l in texto.splitlines() if l.startswith("# ")),
            ruta.stem,
        )
        documentos.append(Document(
            page_content=texto,
            metadata={"source": ruta.name, "titulo": titulo, "file_hash": hash_texto(texto)},
        ))

    if not documentos:
        raise ValueError(f"No se encontraron archivos .txt/.md en {data_dir}")
    log.info(f"{len(documentos)} documentos cargados desde {data_dir.name}/")
    return documentos


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
def _crear_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive Character Splitting medido en tokens.

    La jerarquía de separadores intenta cortar primero por secciones Markdown (## ),
    después por párrafos, líneas y oraciones; los caracteres sueltos son el último recurso.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.TOKENIZER_ENCODING,
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def _seccion_de(chunk: str, texto: str, posicion: int) -> str:
    """Sección del chunk: su propio encabezado si empieza con uno; si no, el último
    encabezado ## / ### que aparece antes de 'posicion' en el documento."""
    propio = re.match(r"#{2,3} (.+)", chunk.lstrip())
    if propio:
        return propio.group(1).strip()
    encabezados = re.findall(r"^#{2,3} (.+)$", texto[:posicion], flags=re.MULTILINE)
    return encabezados[-1].strip() if encabezados else "Introducción"


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Divide cada documento en chunks y les agrega metadatos para citar la fuente.

    Cada chunk se antepone con el título del documento y su sección ("contextual chunk
    header"): así un fragmento que quedó sin el encabezado sigue siendo comprensible
    para el modelo de embeddings y para el LLM.
    """
    splitter = _crear_splitter()
    chunks: list[Document] = []

    for doc in documentos:
        partes = splitter.split_documents([doc])
        for i, parte in enumerate(partes):
            seccion = _seccion_de(parte.page_content, doc.page_content, parte.metadata["start_index"])
            cabecera = f"[{doc.metadata['titulo']} > {seccion}]\n"
            contenido = cabecera + parte.page_content
            parte.page_content = contenido
            parte.metadata.update({
                "chunk_index": i,
                "seccion": seccion,
                "tokens": contar_tokens(contenido),
                # ID determinista: el mismo contenido siempre genera el mismo ID
                "chunk_id": f"{doc.metadata['source']}::{i}::{hash_texto(contenido)[:12]}",
            })
            chunks.append(parte)

        tokens = [p.metadata["tokens"] for p in partes]
        log.info(
            f"  {doc.metadata['source']}: {len(partes)} chunks "
            f"(tokens min={min(tokens)}, max={max(tokens)})"
        )
    _validar_limite_embeddings(chunks)
    return chunks


def _validar_limite_embeddings(chunks: list[Document]) -> None:
    """Avisa si algún chunk supera la ventana del modelo de embeddings (se truncaría).

    tiktoken mide lo que cobra/limita el LLM, pero el modelo de embeddings usa su propio
    tokenizador: se valida también con él.
    """
    modelo = config.get_embeddings()._client
    limite = modelo.max_seq_length
    excedidos = [
        n for c in chunks
        if (n := len(modelo.tokenizer("passage: " + c.page_content)["input_ids"])) > limite
    ]
    if excedidos:
        log.warning(
            f"  {len(excedidos)}/{len(chunks)} chunks superan los {limite} tokens del modelo "
            f"de embeddings (máx. {max(excedidos)}): se trunca su final, que igual queda "
            "cubierto por el overlap del chunk siguiente"
        )
    else:
        log.info(f"  Todos los chunks entran en la ventana de {limite} tokens del modelo de embeddings")


# --------------------------------------------------------------------------- #
# 3. Persistencia en ChromaDB
# --------------------------------------------------------------------------- #
def get_vectorstore() -> Chroma:
    """Abre (o crea) la colección persistente en disco.

    El nombre del modelo de embeddings queda guardado en los metadatos de la colección,
    y se verifica en cada apertura para no consultar con un modelo distinto al de indexación.
    """
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=config.get_embeddings(),
        persist_directory=str(config.VECTORSTORE_DIR),
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": config.EMBEDDING_MODEL,
        },
    )
    modelo_guardado = (vectorstore._collection.metadata or {}).get("embedding_model")
    if modelo_guardado and modelo_guardado != config.EMBEDDING_MODEL:
        raise EmbeddingsIncompatiblesError(
            f"La colección '{config.COLLECTION_NAME}' se indexó con '{modelo_guardado}' "
            f"pero el modelo configurado es '{config.EMBEDDING_MODEL}'. "
            "Ejecutá `python ingest.py --reset` para reindexar."
        )
    return vectorstore


def _estado_actual(vectorstore: Chroma) -> dict[str, dict]:
    """Agrupa lo que ya está indexado por archivo: {source: {"hash": ..., "ids": [...]}}."""
    existentes = vectorstore.get(include=["metadatas"])
    estado: dict[str, dict] = {}
    for id_, meta in zip(existentes["ids"], existentes["metadatas"]):
        info = estado.setdefault(meta["source"], {"hash": meta.get("file_hash"), "ids": []})
        info["ids"].append(id_)
    return estado


def ingest_documents(data_dir: Path = config.DATA_DIR, reset: bool = False) -> Chroma:
    """Pipeline de ingesta completo, idempotente e incremental.

    - Si un archivo ya está indexado y no cambió (mismo hash), no se vuelve a embeber.
    - Si cambió, se borran sus chunks viejos y se insertan los nuevos.
    - Si se eliminó de /data, se borran sus chunks de la colección.
    """
    if reset and config.VECTORSTORE_DIR.exists():
        # Se borra con el cliente nativo: no pasa por la verificación de embeddings,
        # que es justamente lo que el reset viene a corregir.
        cliente = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
        if config.COLLECTION_NAME in [c.name for c in cliente.list_collections()]:
            log.info("--reset: se elimina la colección existente")
            cliente.delete_collection(config.COLLECTION_NAME)

    vectorstore = get_vectorstore()
    estado = _estado_actual(vectorstore)
    documentos = cargar_documentos(data_dir)

    a_indexar = []
    for doc in documentos:
        fuente = doc.metadata["source"]
        previo = estado.get(fuente)
        if previo and previo["hash"] == doc.metadata["file_hash"]:
            log.info(f"  {fuente}: sin cambios, se omite")
            continue
        if previo:
            log.info(f"  {fuente}: modificado, se reemplazan {len(previo['ids'])} chunks")
            vectorstore.delete(ids=previo["ids"])
        a_indexar.append(doc)

    # Archivos que ya no están en /data
    fuentes_actuales = {d.metadata["source"] for d in documentos}
    for fuente, info in estado.items():
        if fuente not in fuentes_actuales:
            log.info(f"  {fuente}: ya no existe en /data, se eliminan sus chunks")
            vectorstore.delete(ids=info["ids"])

    if a_indexar:
        chunks = fragmentar(a_indexar)
        vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        log.info(f"Indexados {len(chunks)} chunks nuevos")
    else:
        log.info("La base vectorial ya está al día: no hace falta reindexar")

    log.info(
        f"Colección '{config.COLLECTION_NAME}': {vectorstore._collection.count()} chunks "
        f"en {config.VECTORSTORE_DIR.name}/"
    )
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos en ChromaDB")
    parser.add_argument("--reset", action="store_true", help="borra la colección y reindexa todo")
    args = parser.parse_args()
    ingest_documents(reset=args.reset)
