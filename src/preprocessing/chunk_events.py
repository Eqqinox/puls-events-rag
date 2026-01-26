"""
Script de découpage des textes en chunks pour la vectorisation.

Ce script découpe les textes préparés en chunks de taille appropriée
pour l'embedding avec Mistral, en utilisant LangChain.

Stratégie de chunking :
- Utilisation de RecursiveCharacterTextSplitter de LangChain
- Séparateurs par défaut : paragraphes, lignes, phrases, espaces
- Textes courts (< chunk_size) : 1 chunk = 1 événement
- Textes longs (>= chunk_size) : découpage avec overlap

Usage:
    python src/preprocessing/chunk_events.py

Entrée: src/data/processed/events_for_vectorization.json
Sortie: src/data/processed/events_chunked.json
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "events_for_vectorization.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "events_chunked.json"


# Paramètres de chunking
CHUNK_SIZE = 800  # Taille cible des chunks en caractères
CHUNK_OVERLAP = 100  # Chevauchement entre chunks
MIN_CHUNK_SIZE = 100  # Taille minimale d'un chunk

# Séparateurs pour le découpage (du plus prioritaire au moins prioritaire)
# Priorité : double saut de ligne > saut de ligne > point + espace > espace
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# Instance du text splitter LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=SEPARATORS,
    keep_separator=True,
    strip_whitespace=True
)


def split_text_into_chunks(
    text: str,
    min_size: int = MIN_CHUNK_SIZE
) -> list[str]:
    """
    Découpe un texte en chunks avec overlap en utilisant LangChain.
    
    Args:
        text: Texte à découper
        min_size: Taille minimale d'un chunk
        
    Returns:
        Liste des chunks
    """
    if not text:
        return []
    
    # Utilise LangChain pour découper le texte
    chunks = text_splitter.split_text(text)
    
    # Filtre les chunks trop petits
    filtered_chunks = [chunk for chunk in chunks if len(chunk) >= min_size]
    
    # Si tous les chunks sont filtrés mais le texte original est valide,
    # on retourne le texte original comme unique chunk
    if not filtered_chunks and len(text.strip()) >= min_size:
        return [text.strip()]
    
    return filtered_chunks


def chunk_event(event: dict) -> list[dict]:
    """
    Découpe un événement en un ou plusieurs chunks.
    
    Args:
        event: Événement avec text_for_embedding et metadata
        
    Returns:
        Liste de chunks avec leurs métadonnées
    """
    text = event.get("text_for_embedding", "")
    metadata = event.get("metadata", {})
    event_id = event.get("id", "")
    
    # Découpe le texte avec LangChain
    text_chunks = split_text_into_chunks(text)
    
    # Crée un document par chunk
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_doc = {
            "chunk_id": f"{event_id}_{i}" if len(text_chunks) > 1 else event_id,
            "event_id": event_id,
            "chunk_index": i,
            "total_chunks": len(text_chunks),
            "text": chunk_text,
            "metadata": metadata
        }
        chunks.append(chunk_doc)
    
    return chunks


def load_prepared_events(path: Path) -> tuple[list[dict], dict]:
    """
    Charge les événements préparés depuis le fichier JSON.
    
    Args:
        path: Chemin vers le fichier JSON
        
    Returns:
        Tuple (liste des événements, métadonnées)
    """
    logger.info(f"Chargement des données depuis {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    events = data.get("events", [])
    metadata = data.get("metadata", {})
    
    logger.info(f"Chargé {len(events)} événements")
    
    return events, metadata


def save_chunked_events(chunks: list[dict], metadata: dict, stats: dict, path: Path) -> None:
    """
    Sauvegarde les chunks dans un fichier JSON.
    
    Args:
        chunks: Liste des chunks
        metadata: Métadonnées des étapes précédentes
        stats: Statistiques de chunking
        path: Chemin de sortie
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    chunking_metadata = {
        "source_metadata": metadata,
        "chunking": {
            "chunked_at": datetime.utcnow().isoformat() + "Z",
            "method": "langchain_recursive_character_text_splitter",
            "parameters": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "min_chunk_size": MIN_CHUNK_SIZE,
                "separators": SEPARATORS
            },
            "statistics": stats
        }
    }
    
    output_data = {
        "metadata": chunking_metadata,
        "chunks": chunks
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Données sauvegardées dans {path}")


def main():
    """Fonction principale de chunking."""
    logger.info("=" * 60)
    logger.info("DÉCOUPAGE DES TEXTES EN CHUNKS (LangChain)")
    logger.info("=" * 60)
    
    # Charge les données préparées
    events, metadata = load_prepared_events(INPUT_PATH)
    
    # Statistiques
    stats = {
        "total_events": len(events),
        "total_chunks": 0,
        "events_single_chunk": 0,
        "events_multiple_chunks": 0,
        "max_chunks_per_event": 0,
        "avg_chunk_length": 0,
        "min_chunk_length": float("inf"),
        "max_chunk_length": 0
    }
    
    all_chunks = []
    chunk_lengths = []
    
    for event in events:
        chunks = chunk_event(event)
        all_chunks.extend(chunks)
        
        # Met à jour les statistiques
        num_chunks = len(chunks)
        stats["total_chunks"] += num_chunks
        
        if num_chunks == 1:
            stats["events_single_chunk"] += 1
        else:
            stats["events_multiple_chunks"] += 1
        
        if num_chunks > stats["max_chunks_per_event"]:
            stats["max_chunks_per_event"] = num_chunks
        
        for chunk in chunks:
            length = len(chunk["text"])
            chunk_lengths.append(length)
            if length < stats["min_chunk_length"]:
                stats["min_chunk_length"] = length
            if length > stats["max_chunk_length"]:
                stats["max_chunk_length"] = length
    
    # Calcule la moyenne
    if chunk_lengths:
        stats["avg_chunk_length"] = round(sum(chunk_lengths) / len(chunk_lengths), 2)
    
    if stats["min_chunk_length"] == float("inf"):
        stats["min_chunk_length"] = 0
    
    # Affiche les statistiques
    logger.info("-" * 60)
    logger.info("STATISTIQUES DE CHUNKING")
    logger.info("-" * 60)
    logger.info(f"Événements traités         : {stats['total_events']}")
    logger.info(f"Total chunks générés       : {stats['total_chunks']}")
    logger.info(f"Événements (1 chunk)       : {stats['events_single_chunk']}")
    logger.info(f"Événements (multi-chunks)  : {stats['events_multiple_chunks']}")
    logger.info(f"Max chunks par événement   : {stats['max_chunks_per_event']}")
    logger.info(f"Longueur moyenne chunk     : {stats['avg_chunk_length']} car.")
    logger.info(f"Longueur min chunk         : {stats['min_chunk_length']} car.")
    logger.info(f"Longueur max chunk         : {stats['max_chunk_length']} car.")
    logger.info("-" * 60)
    
    # Affiche un exemple
    if all_chunks:
        logger.info("EXEMPLE DE CHUNK")
        logger.info("-" * 60)
        example = all_chunks[0]
        logger.info(f"Chunk ID: {example['chunk_id']}")
        logger.info(f"Event ID: {example['event_id']}")
        logger.info(f"Index: {example['chunk_index']}/{example['total_chunks']}")
        logger.info(f"Texte ({len(example['text'])} car.):")
        logger.info(example['text'][:300] + "..." if len(example['text']) > 300 else example['text'])
        logger.info("-" * 60)
    
    # Sauvegarde
    save_chunked_events(all_chunks, metadata, stats, OUTPUT_PATH)

    logger.info("Chunking terminé avec succès")

    return 0


if __name__ == "__main__":
    exit(main())