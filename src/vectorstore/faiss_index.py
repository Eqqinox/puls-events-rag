"""
Script de création et gestion de l'index vectoriel Faiss.

Ce script crée un index Faiss à partir des embeddings Mistral générés
et permet la recherche sémantique des événements culturels.

Usage:
    python src/vectorstore/faiss_index.py

Entrées:
    - data/processed/embeddings.npy (15928 vecteurs x 1024 dimensions)
    - data/processed/events_vectorized.json (métadonnées des chunks)

Sorties:
    - data/processed/faiss_index/index.faiss (index Faiss)
    - data/processed/faiss_index/index.pkl (métadonnées)
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "embeddings.npy"
METADATA_PATH = BASE_DIR / "data" / "processed" / "events_vectorized.json"
INDEX_PATH = BASE_DIR / "data" / "processed" / "faiss_index"


def load_embeddings_and_metadata(
    embeddings_path: Path = EMBEDDINGS_PATH,
    metadata_path: Path = METADATA_PATH
) -> tuple[np.ndarray, list[dict]]:
    """
    Charge les embeddings et les métadonnées depuis les fichiers.

    Args:
        embeddings_path: Chemin vers le fichier numpy des embeddings
        metadata_path: Chemin vers le fichier JSON des métadonnées

    Returns:
        Tuple (embeddings numpy array, liste des chunks avec métadonnées)

    Raises:
        FileNotFoundError: Si les fichiers n'existent pas
        ValueError: Si les données ne sont pas cohérentes
    """
    logger.info(f"Chargement des embeddings depuis {embeddings_path}")

    # Vérifie l'existence des fichiers
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Fichier embeddings introuvable: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Fichier métadonnées introuvable: {metadata_path}")

    # Charge les embeddings
    embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings chargés: shape {embeddings.shape}")

    # Charge les métadonnées
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    logger.info(f"Métadonnées chargées: {len(chunks)} chunks")

    # Valide la cohérence
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Incohérence: {len(embeddings)} embeddings "
            f"mais {len(chunks)} chunks de métadonnées"
        )

    # Valide la dimension
    if embeddings.shape[1] != 1024:
        raise ValueError(
            f"Dimension incorrecte: {embeddings.shape[1]}, attendu 1024"
        )

    logger.info("Validation réussie: embeddings et métadonnées cohérents")

    return embeddings, chunks


def create_faiss_index(
    embeddings: np.ndarray,
    chunks: list[dict],
    embedding_model: MistralAIEmbeddings
) -> FAISS:
    """
    Crée un index Faiss avec LangChain à partir des embeddings et métadonnées.

    Args:
        embeddings: Matrice numpy des embeddings (n_vectors, dimension)
        chunks: Liste des chunks avec leurs métadonnées
        embedding_model: Modèle d'embeddings Mistral pour les futures requêtes

    Returns:
        Instance FAISS de LangChain prête à l'emploi
    """
    logger.info("Création de l'index Faiss avec LangChain...")

    # Prépare les documents LangChain
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "event_id": chunk["event_id"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                **chunk["metadata"]
            }
        )
        documents.append(doc)

    logger.info(f"Création de {len(documents)} documents LangChain")

    # Convertit les embeddings en liste pour LangChain
    embeddings_list = embeddings.tolist()

    # Crée les paires (texte, embedding) pour LangChain
    text_embeddings = list(zip(
        [doc.page_content for doc in documents],
        embeddings_list
    ))

    # Crée l'index FAISS via LangChain
    logger.info("Création de l'index vectoriel...")
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_model,
        metadatas=[doc.metadata for doc in documents]
    )

    logger.info("Index Faiss créé avec succès")

    return vectorstore


def build_and_save_faiss_index(
    embeddings_path: Path = EMBEDDINGS_PATH,
    metadata_path: Path = METADATA_PATH,
    output_path: Path = INDEX_PATH,
    api_key: Optional[str] = None
) -> dict:
    """
    Pipeline complet: charge les données, crée l'index et sauvegarde.

    Args:
        embeddings_path: Chemin vers les embeddings numpy
        metadata_path: Chemin vers les métadonnées JSON
        output_path: Chemin de sortie pour l'index
        api_key: Clé API Mistral (optionnel, utilise .env par défaut)

    Returns:
        Dictionnaire avec les statistiques de création
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("CREATION DE L'INDEX FAISS")
    logger.info("=" * 60)

    # Configure la clé API Mistral
    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key
    elif not os.getenv("MISTRAL_API_KEY"):
        raise ValueError(
            "MISTRAL_API_KEY non trouvée. "
            "Définissez-la dans .env ou passez-la en paramètre."
        )

    # Initialise le modèle d'embeddings
    logger.info("Initialisation du modèle Mistral Embeddings...")
    embedding_model = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    # Charge les données
    embeddings, chunks = load_embeddings_and_metadata(
        embeddings_path,
        metadata_path
    )

    # Crée l'index
    vectorstore = create_faiss_index(embeddings, chunks, embedding_model)

    # Crée le dossier de sortie
    output_path.mkdir(parents=True, exist_ok=True)

    # Sauvegarde l'index
    logger.info(f"Sauvegarde de l'index dans {output_path}")
    vectorstore.save_local(str(output_path))

    # Calcule les statistiques
    end_time = time.time()
    build_time = end_time - start_time

    # Taille des fichiers
    index_file = output_path / "index.faiss"
    pkl_file = output_path / "index.pkl"

    index_size = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0
    pkl_size = pkl_file.stat().st_size / (1024 * 1024) if pkl_file.exists() else 0

    stats = {
        "total_vectors": len(embeddings),
        "embedding_dimension": embeddings.shape[1],
        "index_type": "IndexFlatL2",
        "index_size_mb": round(index_size, 2),
        "metadata_size_mb": round(pkl_size, 2),
        "total_size_mb": round(index_size + pkl_size, 2),
        "build_time_seconds": round(build_time, 2),
        "index_path": str(output_path),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    # Affiche les statistiques
    logger.info("-" * 60)
    logger.info("STATISTIQUES DE L'INDEX")
    logger.info("-" * 60)
    logger.info(f"Nombre de vecteurs       : {stats['total_vectors']:,}")
    logger.info(f"Dimension                : {stats['embedding_dimension']}")
    logger.info(f"Type d'index             : {stats['index_type']}")
    logger.info(f"Taille index             : {stats['index_size_mb']} MB")
    logger.info(f"Taille métadonnées       : {stats['metadata_size_mb']} MB")
    logger.info(f"Taille totale            : {stats['total_size_mb']} MB")
    logger.info(f"Temps de création        : {stats['build_time_seconds']} secondes")
    logger.info(f"Chemin de sauvegarde     : {stats['index_path']}")
    logger.info("-" * 60)

    logger.info("Index Faiss créé et sauvegardé avec succès")

    return stats


def load_faiss_index(
    index_path: Path = INDEX_PATH,
    api_key: Optional[str] = None
) -> FAISS:
    """
    Charge un index Faiss existant depuis le disque.

    Args:
        index_path: Chemin vers le dossier contenant l'index
        api_key: Clé API Mistral (optionnel, utilise .env par défaut)

    Returns:
        Instance FAISS chargée et prête à l'emploi

    Raises:
        FileNotFoundError: Si l'index n'existe pas
    """
    logger.info(f"Chargement de l'index Faiss depuis {index_path}")

    # Vérifie l'existence
    if not index_path.exists():
        raise FileNotFoundError(f"Index introuvable: {index_path}")

    # Configure la clé API
    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key
    elif not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY non trouvée")

    # Initialise le modèle d'embeddings
    embedding_model = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    # Charge l'index
    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    logger.info("Index Faiss chargé avec succès")

    return vectorstore


def test_search(
    vectorstore: FAISS,
    query: str,
    k: int = 5
) -> list[dict]:
    """
    Teste la recherche sémantique avec une requête exemple.

    Args:
        vectorstore: Instance FAISS chargée
        query: Requête de recherche en langage naturel
        k: Nombre de résultats à retourner

    Returns:
        Liste des résultats avec texte, métadonnées et score
    """
    logger.info(f"Recherche: '{query}' (top {k})")

    # Effectue la recherche avec scores
    results = vectorstore.similarity_search_with_score(query, k=k)

    # Formate les résultats
    formatted_results = []
    for i, (doc, score) in enumerate(results, 1):
        result = {
            "rank": i,
            "score": float(score),
            "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": {
                "event_id": doc.metadata.get("event_id"),
                "title": doc.metadata.get("title"),
                "date_range": doc.metadata.get("date_range"),
                "city": doc.metadata.get("city"),
                "url": doc.metadata.get("url")
            }
        }
        formatted_results.append(result)

        logger.info(f"  [{i}] Score: {score:.4f} - {doc.metadata.get('title', 'Sans titre')}")

    return formatted_results


def main():
    """Point d'entrée principal du script."""
    try:
        # Crée l'index
        stats = build_and_save_faiss_index()

        logger.info("\n" + "=" * 60)
        logger.info("TEST DE RECHERCHE")
        logger.info("=" * 60)

        # Charge l'index pour test
        vectorstore = load_faiss_index()

        # Test avec une requête exemple
        test_queries = [
            "concert de jazz à Paris ce weekend",
            "exposition d'art contemporain",
            "théâtre pour enfants"
        ]

        for query in test_queries:
            logger.info(f"\nRequête: '{query}'")
            results = test_search(vectorstore, query, k=3)
            logger.info(f"  -> {len(results)} résultats trouvés\n")

        logger.info("=" * 60)
        logger.info("Script terminé avec succès")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
