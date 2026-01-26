"""
Script de vectorisation des chunks avec Mistral Embeddings.

Ce script génère les embeddings pour chaque chunk en utilisant
l'API Mistral Embeddings.

Usage:
    python src/preprocessing/vectorize_events.py

Entrée: src/data/processed/events_chunked.json
Sortie: src/data/processed/events_vectorized.json
        src/data/processed/embeddings.npy (vecteurs numpy)
"""

import json
import logging
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral

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
INPUT_PATH = BASE_DIR / "data" / "processed" / "events_chunked.json"
OUTPUT_JSON_PATH = BASE_DIR / "data" / "processed" / "events_vectorized.json"
OUTPUT_NPY_PATH = BASE_DIR / "data" / "processed" / "embeddings.npy"
CHECKPOINT_PATH = BASE_DIR / "data" / "processed" / "vectorization_checkpoint.json"


# Configuration Mistral
MISTRAL_MODEL = "mistral-embed"
EMBEDDING_DIMENSION = 1024
BATCH_SIZE = 50  # Nombre de textes par batch
RATE_LIMIT_DELAY = 1.0  # Délai entre les batches (secondes) - augmenté
RATE_LIMIT_429_DELAY = 30  # Délai après une erreur 429 (secondes)
MAX_RETRIES = 5  # Nombre maximum de tentatives


def get_mistral_client() -> Mistral:
    """
    Crée et retourne un client Mistral.
    
    Returns:
        Client Mistral configuré
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY non trouvée dans les variables d'environnement")
    
    return Mistral(api_key=api_key)


def embed_batch(client: Mistral, texts: list[str], retry_count: int = MAX_RETRIES) -> list[list[float]]:
    """
    Génère les embeddings pour un batch de textes.
    
    Args:
        client: Client Mistral
        texts: Liste de textes à vectoriser
        retry_count: Nombre de tentatives en cas d'erreur
        
    Returns:
        Liste des vecteurs d'embedding
    """
    for attempt in range(retry_count):
        try:
            response = client.embeddings.create(
                model=MISTRAL_MODEL,
                inputs=texts
            )
            
            # Extrait les embeddings dans l'ordre
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "capacity" in error_str.lower()
            
            logger.warning(f"Erreur batch (tentative {attempt + 1}/{retry_count}): {e}")
            
            if attempt < retry_count - 1:
                if is_rate_limit:
                    # Délai plus long pour les erreurs de rate limiting
                    wait_time = RATE_LIMIT_429_DELAY * (attempt + 1)
                    logger.info(f"Rate limit atteint. Pause de {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)
                else:
                    # Backoff exponentiel pour les autres erreurs
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
            else:
                raise


def load_chunked_events(path: Path) -> tuple[list[dict], dict]:
    """
    Charge les chunks depuis le fichier JSON.
    
    Args:
        path: Chemin vers le fichier JSON
        
    Returns:
        Tuple (liste des chunks, métadonnées)
    """
    logger.info(f"Chargement des données depuis {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    metadata = data.get("metadata", {})
    
    logger.info(f"Chargé {len(chunks)} chunks")
    
    return chunks, metadata


def load_checkpoint(path: Path) -> tuple[int, list[list[float]]]:
    """
    Charge un checkpoint de vectorisation si disponible.
    
    Args:
        path: Chemin vers le fichier de checkpoint
        
    Returns:
        Tuple (index de reprise, embeddings déjà générés)
    """
    if not path.exists():
        return 0, []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        
        start_index = checkpoint.get("last_processed_index", 0)
        embeddings = checkpoint.get("embeddings", [])
        
        logger.info(f"Checkpoint trouvé: reprise à l'index {start_index} ({len(embeddings)} embeddings)")
        return start_index, embeddings
        
    except Exception as e:
        logger.warning(f"Erreur lors du chargement du checkpoint: {e}")
        return 0, []


def save_checkpoint(path: Path, index: int, embeddings: list[list[float]]) -> None:
    """
    Sauvegarde un checkpoint de vectorisation.
    
    Args:
        path: Chemin vers le fichier de checkpoint
        index: Dernier index traité
        embeddings: Liste des embeddings générés
    """
    checkpoint = {
        "last_processed_index": index,
        "embeddings": embeddings,
        "saved_at": datetime.utcnow().isoformat() + "Z"
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f)


def delete_checkpoint(path: Path) -> None:
    """
    Supprime le fichier de checkpoint après succès.
    
    Args:
        path: Chemin vers le fichier de checkpoint
    """
    if path.exists():
        path.unlink()
        logger.info("Checkpoint supprimé")


def save_vectorized_data(
    chunks: list[dict],
    embeddings: np.ndarray,
    metadata: dict,
    stats: dict,
    json_path: Path,
    npy_path: Path
) -> None:
    """
    Sauvegarde les données vectorisées.
    
    Args:
        chunks: Liste des chunks avec métadonnées
        embeddings: Matrice numpy des embeddings
        metadata: Métadonnées des étapes précédentes
        stats: Statistiques de vectorisation
        json_path: Chemin pour le fichier JSON
        npy_path: Chemin pour le fichier numpy
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Métadonnées de vectorisation
    vectorization_metadata = {
        "source_metadata": metadata,
        "vectorization": {
            "vectorized_at": datetime.utcnow().isoformat() + "Z",
            "model": MISTRAL_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "statistics": stats,
            "files": {
                "json": str(json_path.name),
                "embeddings": str(npy_path.name)
            }
        }
    }
    
    # Prépare les chunks sans les embeddings (stockés séparément en numpy)
    chunks_for_json = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "index": i,
            "chunk_id": chunk["chunk_id"],
            "event_id": chunk["event_id"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        }
        chunks_for_json.append(chunk_data)
    
    output_data = {
        "metadata": vectorization_metadata,
        "chunks": chunks_for_json
    }
    
    # Sauvegarde JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Métadonnées sauvegardées dans {json_path}")
    
    # Sauvegarde numpy
    np.save(npy_path, embeddings)
    logger.info(f"Embeddings sauvegardés dans {npy_path}")


def main():
    """Fonction principale de vectorisation."""
    logger.info("=" * 60)
    logger.info("VECTORISATION DES CHUNKS AVEC MISTRAL")
    logger.info("=" * 60)
    
    # Charge les chunks
    chunks, metadata = load_chunked_events(INPUT_PATH)
    total_chunks = len(chunks)
    
    # Vérifie s'il y a un checkpoint
    start_index, existing_embeddings = load_checkpoint(CHECKPOINT_PATH)
    
    # Initialise le client Mistral
    logger.info("Initialisation du client Mistral...")
    client = get_mistral_client()
    
    # Statistiques
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
    stats = {
        "total_chunks": total_chunks,
        "total_batches": total_batches,
        "batch_size": BATCH_SIZE,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "model": MISTRAL_MODEL
    }
    
    if start_index > 0:
        logger.info(f"Reprise de la vectorisation à partir de l'index {start_index}")
    
    logger.info(f"Vectorisation de {total_chunks} chunks en {total_batches} batches...")
    
    # Génère les embeddings par batch
    all_embeddings = existing_embeddings.copy()
    start_time = time.time()
    checkpoint_interval = 50  # Sauvegarde checkpoint tous les 50 batches
    
    try:
        for i in range(start_index, total_chunks, BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch_chunks = chunks[i:i + BATCH_SIZE]
            batch_texts = [chunk["text"] for chunk in batch_chunks]
            
            # Log de progression
            if batch_num % 10 == 0 or batch_num == 1 or i == start_index:
                progress = (i / total_chunks) * 100
                logger.info(f"Batch {batch_num}/{total_batches} ({progress:.1f}%)")
            
            # Génère les embeddings
            embeddings = embed_batch(client, batch_texts)
            all_embeddings.extend(embeddings)
            
            # Sauvegarde checkpoint périodique
            if batch_num % checkpoint_interval == 0:
                save_checkpoint(CHECKPOINT_PATH, i + BATCH_SIZE, all_embeddings)
                logger.info(f"Checkpoint sauvegardé à l'index {i + BATCH_SIZE}")
            
            # Rate limiting
            if i + BATCH_SIZE < total_chunks:
                time.sleep(RATE_LIMIT_DELAY)
                
    except Exception as e:
        # En cas d'erreur, sauvegarde le checkpoint avant de quitter
        logger.error(f"Erreur lors de la vectorisation: {e}")
        save_checkpoint(CHECKPOINT_PATH, i, all_embeddings)
        logger.info(f"Checkpoint de récupération sauvegardé à l'index {i}")
        raise
    
    elapsed_time = time.time() - start_time
    stats["processing_time_seconds"] = round(elapsed_time, 2)
    stats["chunks_per_second"] = round((total_chunks - start_index) / elapsed_time, 2) if elapsed_time > 0 else 0
    
    # Convertit en numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    
    # Affiche les statistiques
    logger.info("-" * 60)
    logger.info("STATISTIQUES DE VECTORISATION")
    logger.info("-" * 60)
    logger.info(f"Chunks vectorisés          : {total_chunks}")
    logger.info(f"Dimension des embeddings   : {EMBEDDING_DIMENSION}")
    logger.info(f"Taille matrice             : {embeddings_array.shape}")
    logger.info(f"Temps de traitement        : {elapsed_time:.1f}s")
    logger.info(f"Vitesse                    : {stats['chunks_per_second']} chunks/s")
    logger.info(f"Mémoire embeddings         : {embeddings_array.nbytes / 1024 / 1024:.2f} MB")
    logger.info("-" * 60)
    
    # Sauvegarde
    save_vectorized_data(
        chunks,
        embeddings_array,
        metadata,
        stats,
        OUTPUT_JSON_PATH,
        OUTPUT_NPY_PATH
    )
    
    # Supprime le checkpoint après succès
    delete_checkpoint(CHECKPOINT_PATH)

    logger.info("Vectorisation terminée avec succès")

    return 0


if __name__ == "__main__":
    exit(main())