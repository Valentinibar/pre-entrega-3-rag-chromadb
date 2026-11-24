"""Contratos de datos del sistema RAG.

- RespuestaLLM: lo que el LLM está obligado a devolver (lo parsea PydanticOutputParser).
- RespuestaRAG: la respuesta final de get_rag_response(), con las referencias ya
  verificadas contra los fragmentos que realmente se recuperaron de ChromaDB.
"""

from pydantic import BaseModel, Field, model_validator

NO_LO_SE = "No lo sé. No tengo acceso a esa información en la documentación disponible."


class RespuestaLLM(BaseModel):
    """Salida estructurada que debe generar el modelo de lenguaje."""

    encontrado: bool = Field(
        description="true si el CONTEXTO contiene la información para responder; false si no.",
    )
    respuesta: str = Field(
        min_length=1,
        description=(
            "Respuesta en español basada únicamente en el CONTEXTO. "
            f'Si encontrado es false, debe ser exactamente: "{NO_LO_SE}"'
        ),
    )
    fragmentos_usados: list[int] = Field(
        default_factory=list,
        description=(
            "Números de los fragmentos del CONTEXTO usados para responder (ej. [1, 3]). "
            "Lista vacía si encontrado es false."
        ),
    )

    @model_validator(mode="after")
    def coherencia(self) -> "RespuestaLLM":
        if self.encontrado and not self.fragmentos_usados:
            raise ValueError("si encontrado=true hay que citar al menos un fragmento")
        if not self.encontrado:
            # Normalizamos: sin información no hay respuesta inventada ni citas
            self.respuesta = NO_LO_SE
            self.fragmentos_usados = []
        self.fragmentos_usados = sorted(set(self.fragmentos_usados))
        return self


class Referencia(BaseModel):
    """Fuente concreta (archivo + sección) que respalda la respuesta."""

    fuente: str = Field(description="Nombre del archivo de /data")
    seccion: str = Field(description="Sección del documento donde está la información")
    chunk_index: int = Field(ge=0, description="Posición del fragmento dentro del archivo")
    relevancia: float = Field(ge=-1, le=1, description="Similitud coseno con la pregunta")
    extracto: str = Field(description="Primeros caracteres del fragmento, para trazabilidad")


class RespuestaRAG(BaseModel):
    """Respuesta final del sistema: texto + referencias trazables."""

    pregunta: str
    respuesta: str
    encontrado: bool
    referencias: list[Referencia] = Field(default_factory=list)

    @model_validator(mode="after")
    def referencias_obligatorias(self) -> "RespuestaRAG":
        if self.encontrado and not self.referencias:
            raise ValueError("una respuesta basada en el contexto debe incluir referencias")
        if not self.encontrado and self.referencias:
            raise ValueError("una respuesta 'No lo sé' no puede tener referencias")
        return self
