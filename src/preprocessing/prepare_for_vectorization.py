"""
Script de préparation des données pour la vectorisation.

Ce script structure les événements nettoyés pour l'indexation dans Faiss :
1. Création du champ text_for_embedding combinant les informations clés
2. Structuration des métadonnées pour la recherche
3. Sélection des champs utiles uniquement

Usage:
    python src/preprocessing/prepare_for_vectorization.py

Entrée: src/data/processed/events_cleaned.json
Sortie: src/data/processed/events_for_vectorization.json
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "events_cleaned.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "events_for_vectorization.json"


def create_text_for_embedding(event: dict) -> str:
    """
    Crée le texte combiné pour l'embedding.
    
    Le texte est structuré pour maximiser la pertinence sémantique :
    - Titre (information principale)
    - Description courte
    - Description longue (si disponible)
    - Localisation (lieu, ville, département)
    - Date
    - Mots-clés (si disponibles)
    
    Args:
        event: Dictionnaire de l'événement nettoyé
        
    Returns:
        Texte combiné pour l'embedding
    """
    parts = []
    
    # Titre
    title = (event.get("title_fr") or "").strip()
    if title:
        parts.append(f"Titre: {title}")
    
    # Description courte
    description = (event.get("description_fr") or "").strip()
    if description:
        parts.append(f"Description: {description}")
    
    # Description longue (tronquée si trop longue pour éviter les embeddings trop longs)
    long_desc = event.get("longdescription_fr") or ""
    if isinstance(long_desc, str) and long_desc.strip():
        long_desc = long_desc.strip()
        # Limite à 1000 caractères pour éviter des textes trop longs
        if len(long_desc) > 1000:
            long_desc = long_desc[:1000] + "..."
        if long_desc and long_desc != description:
            parts.append(f"Détails: {long_desc}")
    
    # Localisation
    location_parts = []
    location_name = (event.get("location_name") or "").strip()
    city = (event.get("location_city") or "").strip()
    department = (event.get("location_department") or "").strip()
    
    if location_name:
        location_parts.append(location_name)
    if city:
        location_parts.append(city)
    if department:
        location_parts.append(department)
    
    if location_parts:
        parts.append(f"Lieu: {', '.join(location_parts)}")
    
    # Date
    daterange = (event.get("daterange_fr") or "").strip()
    if daterange:
        parts.append(f"Date: {daterange}")
    
    # Mots-clés
    keywords = event.get("keywords_fr") or ""
    if isinstance(keywords, str) and keywords.strip():
        parts.append(f"Mots-clés: {keywords.strip()}")
    
    # Conditions (gratuit, tarif, etc.)
    conditions = event.get("conditions_fr") or ""
    if isinstance(conditions, str) and conditions.strip():
        parts.append(f"Conditions: {conditions.strip()}")
    
    return "\n".join(parts)


def extract_metadata(event: dict) -> dict:
    """
    Extrait les métadonnées utiles pour la recherche et l'affichage.
    
    Args:
        event: Dictionnaire de l'événement nettoyé
        
    Returns:
        Dictionnaire des métadonnées structurées
    """
    return {
        "uid": event.get("uid", ""),
        "title": event.get("title_fr", ""),
        "description": event.get("description_fr", ""),
        "date_range": event.get("daterange_fr", ""),
        "date_start": event.get("firstdate_begin", ""),
        "date_end": event.get("firstdate_end", ""),
        "location_name": event.get("location_name", ""),
        "city": event.get("location_city", ""),
        "department": event.get("location_department", ""),
        "url": event.get("canonicalurl", ""),
        "image": event.get("image", ""),
        "keywords": event.get("keywords_fr", ""),
        "conditions": event.get("conditions_fr", ""),
    }


def prepare_event_for_vectorization(event: dict) -> dict:
    """
    Prépare un événement pour la vectorisation.
    
    Args:
        event: Dictionnaire de l'événement nettoyé
        
    Returns:
        Dictionnaire structuré pour la vectorisation
    """
    return {
        "id": event.get("uid", ""),
        "text_for_embedding": create_text_for_embedding(event),
        "metadata": extract_metadata(event)
    }


def load_cleaned_events(path: Path) -> tuple[list[dict], dict]:
    """
    Charge les événements nettoyés depuis le fichier JSON.
    
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


def save_vectorization_data(events: list[dict], metadata: dict, path: Path) -> None:
    """
    Sauvegarde les données préparées pour la vectorisation.
    
    Args:
        events: Liste des événements préparés
        metadata: Métadonnées des étapes précédentes
        path: Chemin de sortie
    """
    # Crée le dossier si nécessaire
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calcule des statistiques sur les textes
    text_lengths = [len(e["text_for_embedding"]) for e in events]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    min_length = min(text_lengths) if text_lengths else 0
    max_length = max(text_lengths) if text_lengths else 0
    
    # Ajoute les métadonnées de préparation
    preparation_metadata = {
        "source_metadata": metadata,
        "preparation": {
            "prepared_at": datetime.utcnow().isoformat() + "Z",
            "total_events": len(events),
            "text_stats": {
                "avg_length_chars": round(avg_length, 2),
                "min_length_chars": min_length,
                "max_length_chars": max_length
            },
            "fields_in_text": [
                "title_fr",
                "description_fr", 
                "longdescription_fr (tronqué à 1000 car.)",
                "location_name",
                "location_city",
                "location_department",
                "daterange_fr",
                "keywords_fr",
                "conditions_fr"
            ],
            "metadata_fields": [
                "uid",
                "title",
                "description",
                "date_range",
                "date_start",
                "date_end",
                "location_name",
                "city",
                "department",
                "url",
                "image",
                "keywords",
                "conditions"
            ]
        }
    }
    
    output_data = {
        "metadata": preparation_metadata,
        "events": events
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Données sauvegardées dans {path}")


def main():
    """Fonction principale de préparation pour la vectorisation."""
    logger.info("=" * 60)
    logger.info("PREPARATION DES DONNEES POUR LA VECTORISATION")
    logger.info("=" * 60)
    
    # Charge les données nettoyées
    events, metadata = load_cleaned_events(INPUT_PATH)
    
    # Prépare chaque événement
    prepared_events = []
    
    for event in events:
        prepared = prepare_event_for_vectorization(event)
        prepared_events.append(prepared)
    
    # Calcule des statistiques
    text_lengths = [len(e["text_for_embedding"]) for e in prepared_events]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    
    # Compte les événements avec mots-clés
    with_keywords = sum(1 for e in prepared_events if e["metadata"].get("keywords"))
    
    # Affiche les statistiques
    logger.info("-" * 60)
    logger.info("STATISTIQUES DE PREPARATION")
    logger.info("-" * 60)
    logger.info(f"Événements préparés        : {len(prepared_events)}")
    logger.info(f"Longueur moyenne du texte  : {avg_length:.0f} caractères")
    logger.info(f"Longueur min               : {min(text_lengths)} caractères")
    logger.info(f"Longueur max               : {max(text_lengths)} caractères")
    logger.info(f"Événements avec mots-clés  : {with_keywords} ({100*with_keywords/len(prepared_events):.1f}%)")
    logger.info("-" * 60)
    
    # Affiche un exemple
    logger.info("EXEMPLE DE TEXTE POUR EMBEDDING")
    logger.info("-" * 60)
    example = prepared_events[0]
    logger.info(f"ID: {example['id']}")
    logger.info(f"Texte:\n{example['text_for_embedding'][:500]}...")
    logger.info("-" * 60)
    
    # Sauvegarde
    save_vectorization_data(prepared_events, metadata, OUTPUT_PATH)
    
    logger.info("Préparation terminée avec succès")
    
    return prepared_events


if __name__ == "__main__":
    main()