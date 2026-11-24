"""Configuración compartida por la ingesta y la consulta.

Tener una única fuente de verdad para el modelo de embeddings evita el error #1 de RAG:
indexar con un modelo y consultar con otro.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Avisos de librerías que no aportan información en este proyecto
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Silenciamos librerías muy verbosas
for ruidoso in ("httpx", "httpx2", "sentence_transformers", "chromadb", "huggingface_hub"):
    logging.getLogger(ruidoso).setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent

# --- Rutas ---
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
VECTORSTORE_DIR = BASE_DIR / os.getenv("VECTORSTORE_DIR", "vectorstore")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "apuntes_curso")
EXTENSIONES_VALIDAS = {".txt", ".md"}

# --- Chunking (medido en tokens con tiktoken, no en caracteres) ---
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
TOKENIZER_ENCODING = "cl100k_base"

# --- Embeddings: modelo local multilingüe (sin costo, los documentos están en español) ---
# multilingual-e5-small lee hasta 512 tokens por texto, así que embebe el chunk completo.
# (paraphrase-multilingual-MiniLM-L12-v2, por ejemplo, trunca a 128 tokens y perdería el resto.)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# --- Recuperación ---
TOP_K = int(os.getenv("TOP_K", "4"))  # entre 3 y 5 para evitar "Lost in the Middle"
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.70"))  # similitud coseno mínima (calibrada para e5)

# --- LLM ---
LLM_MODEL = os.getenv("LLM_MODEL") or "claude-haiku-5-5"
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

if not 3 <= TOP_K <= 5:
    raise ValueError(f"TOP_K debe estar entre 3 y 5 (actual: {TOP_K})")
if CHUNK_SIZE_TOKENS < 500 or CHUNK_OVERLAP_TOKENS < 50:
    raise ValueError("El chunking debe ser de al menos 500 tokens con 50 de overlap")


@lru_cache(maxsize=1)
def get_embeddings():
    """Devuelve el modelo de embeddings (se carga una sola vez por proceso)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    # Los modelos E5 esperan el prefijo "passage: " en los documentos y "query: " en las
    # preguntas. Los vectores se normalizan para usar similitud coseno.
    def cargar(solo_local: bool):
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": solo_local},
            encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
        )

    # Si el modelo ya está en la caché local se carga sin conectarse a HuggingFace
    # (ahorra varios segundos por ejecución); solo la primera vez se descarga.
    try:
        return cargar(solo_local=True)
    except OSError:
        logging.getLogger("config").info(
            f"Descargando el modelo de embeddings {EMBEDDING_MODEL} (solo la primera vez)..."
        )
        return cargar(solo_local=False)
