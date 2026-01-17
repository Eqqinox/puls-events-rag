"""
Schémas Pydantic pour l'API FastAPI.

Définit les modèles de validation pour les requêtes et réponses
de l'API REST du système RAG Puls-Events.
"""

from typing import List
from pydantic import BaseModel


# ============================================================================
# MODÈLES D'ENTRÉE (REQUÊTES)
# ============================================================================

class AskRequest(BaseModel):
    """Requête pour poser une question au RAG."""
    question: str


# ============================================================================
# MODÈLES DE SORTIE (RÉPONSES)
# ============================================================================

class HealthResponse(BaseModel):
    """Réponse du endpoint /health."""
    status: str


class SourceInfo(BaseModel):
    """Informations sur une source (événement) utilisée pour la réponse."""
    title: str
    location: str
    date: str
    url: str


class AskResponse(BaseModel):
    """Réponse à une question posée au RAG."""
    answer: str
    sources: List[SourceInfo]


class RebuildResponse(BaseModel):
    """Réponse du endpoint /rebuild."""
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Réponse en cas d'erreur."""
    detail: str
