"""
Application FastAPI pour l'API REST du système RAG Puls-Events.

Expose les fonctionnalités du système RAG via des endpoints REST.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

# Ajoute le répertoire racine au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.rag.rag_chain import PulsEventsRAG
from src.api.schemas import (
    AskRequest,
    HealthResponse,
    SourceInfo,
    AskResponse,
    RebuildResponse,
    ErrorResponse
)


# ============================================================================
# CHARGEMENT DU RAG AU DÉMARRAGE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.

    Charge le système RAG au démarrage et le stocke dans app.state
    pour réutilisation dans tous les endpoints.
    """
    # Startup: chargement du RAG (une seule fois)
    app.state.rag = PulsEventsRAG()
    yield
    # Shutdown: nettoyage si nécessaire


# ============================================================================
# APPLICATION FASTAPI
# ============================================================================

app = FastAPI(
    title="Puls-Events RAG API",
    description="API REST pour interroger le système RAG des événements culturels parisiens",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de vérification de l'état de l'API.

    Vérifie que l'API est en ligne et que le système RAG est chargé.

    Returns:
        HealthResponse: {"status": "ok"} si tout est opérationnel

    Raises:
        HTTPException: 503 si le système RAG n'est pas disponible
    """
    # Vérifie que l'instance RAG est chargée
    if not hasattr(app.state, "rag") or app.state.rag is None:
        raise HTTPException(
            status_code=503,
            detail="Le système RAG n'est pas disponible"
        )

    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Endpoint principal pour poser une question au système RAG.

    Reçoit une question, interroge le système RAG et retourne
    une réponse générée avec les sources utilisées.

    Args:
        request: AskRequest contenant la question

    Returns:
        AskResponse: Réponse générée et liste des sources

    Raises:
        HTTPException: 400 si la question est vide, 500 si erreur RAG
    """
    # Vérifie que le RAG est disponible
    if not hasattr(app.state, "rag") or app.state.rag is None:
        raise HTTPException(
            status_code=503,
            detail="Le système RAG n'est pas disponible"
        )

    # Valide que la question n'est pas vide
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide"
        )

    try:
        # Appelle le système RAG
        result = app.state.rag.ask_with_sources(request.question)

        # Formate les sources
        sources = []
        for source_doc in result.get("sources", []):
            metadata = source_doc.metadata
            source_info = SourceInfo(
                title=metadata.get("title", "Sans titre"),
                location=metadata.get("city", "Lieu non spécifié"),
                date=metadata.get("date_range", "Date non spécifiée"),
                url=metadata.get("url", "")
            )
            sources.append(source_info)

        # Retourne la réponse formatée
        return AskResponse(
            answer=result.get("answer", ""),
            sources=sources
        )

    except ValueError as e:
        # Erreur de validation (question vide déjà gérée)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Erreur interne du système RAG
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse: {str(e)}"
        )


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index():
    """
    Endpoint pour reconstruire l'index Faiss et réinitialiser le RAG.

    Recharge complètement le système RAG avec l'index vectoriel actuel.
    Utile après une mise à jour de la base de données d'événements.

    Returns:
        RebuildResponse: Statut et message de l'opération

    Raises:
        HTTPException: 500 si le rechargement échoue
    """
    try:
        # Réinitialise le système RAG
        app.state.rag = PulsEventsRAG()

        return RebuildResponse(
            status="success",
            message="Index Faiss rechargé avec succès"
        )

    except Exception as e:
        # Échec du rechargement
        raise HTTPException(
            status_code=500,
            detail=f"Échec du rechargement de l'index: {str(e)}"
        )
