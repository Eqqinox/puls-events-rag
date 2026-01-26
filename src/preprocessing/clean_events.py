"""
Script de nettoyage des données d'événements Open Agenda.

Ce script applique les règles de nettoyage définies lors de l'exploration :
1. Supprimer les balises HTML de longdescription_fr
2. Normaliser les départements (doublons + casse)
3. Normaliser les villes (casse)
4. Exclure les événements hors Ile-de-France
5. Exclure les événements sans titre ou description
6. Normaliser les mots-clés (casse)

Usage:
    python src/preprocessing/clean_events.py

Entrée: src/data/raw/events_raw.json
Sortie: src/data/processed/events_cleaned.json
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime
from html import unescape
from typing import Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Chemins des fichiers
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "events_raw.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "events_cleaned.json"


# Mapping de normalisation des départements
DEPARTMENT_MAPPING = {
    "Val-D'Oise": "Val-d'Oise",
    "Val-d'oise": "Val-d'Oise",
    "Seine-St-Denis": "Seine-Saint-Denis",
    "Seine-St.-Denis": "Seine-Saint-Denis",
    "seine-saint-denis": "Seine-Saint-Denis",
    "Val-De-Marne": "Val-de-Marne",
    "val-de-marne": "Val-de-Marne",
    "paris": "Paris",
    "PARIS": "Paris",
    "75001": "Paris",
    "75002": "Paris",
    "75003": "Paris",
    "75004": "Paris",
    "75005": "Paris",
    "75006": "Paris",
    "75007": "Paris",
    "75008": "Paris",
    "75009": "Paris",
    "75010": "Paris",
    "75011": "Paris",
    "75012": "Paris",
    "75013": "Paris",
    "75014": "Paris",
    "75015": "Paris",
    "75016": "Paris",
    "75017": "Paris",
    "75018": "Paris",
    "75019": "Paris",
    "75020": "Paris",
}

# Départements valides en Ile-de-France
ILE_DE_FRANCE_DEPARTMENTS = {
    "Paris",
    "Seine-et-Marne",
    "Yvelines",
    "Essonne",
    "Hauts-de-Seine",
    "Seine-Saint-Denis",
    "Val-de-Marne",
    "Val-d'Oise",
}


def remove_html_tags(text: Optional[str]) -> Optional[str]:
    """
    Supprime les balises HTML d'un texte et nettoie les espaces.
    
    Args:
        text: Texte contenant potentiellement du HTML
        
    Returns:
        Texte nettoyé sans balises HTML
    """
    if not text or not isinstance(text, str):
        return text
    
    # Décode les entités HTML (&nbsp;, &amp;, etc.)
    text = unescape(text)
    
    # Remplace les balises <br> et </p> par des sauts de ligne
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
    
    # Supprime toutes les autres balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    
    # Nettoie les espaces multiples et les lignes vides
    text = re.sub(r'[ \t]+', ' ', text)  # Espaces multiples -> un seul
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Lignes vides multiples -> une seule
    text = text.strip()
    
    return text


def normalize_department(department: Optional[str]) -> Optional[str]:
    """
    Normalise le nom du département.
    
    Args:
        department: Nom du département brut
        
    Returns:
        Nom du département normalisé
    """
    if not department or not isinstance(department, str):
        return department
    
    # Applique le mapping si existe
    if department in DEPARTMENT_MAPPING:
        return DEPARTMENT_MAPPING[department]
    
    return department


def normalize_city(city: Optional[str]) -> Optional[str]:
    """
    Normalise le nom de la ville (casse titre).
    
    Args:
        city: Nom de la ville brut
        
    Returns:
        Nom de la ville normalisé
    """
    if not city or not isinstance(city, str):
        return city
    
    # Cas spécial pour PARIS
    if city.upper() == "PARIS":
        return "Paris"
    
    # Garde la casse originale pour les autres villes
    # (évite de casser Saint-Germain-en-Laye, etc.)
    return city


def normalize_keywords(keywords: Optional[str]) -> Optional[str]:
    """
    Normalise les mots-clés (lowercase).
    
    Args:
        keywords: Mots-clés bruts séparés par des virgules
        
    Returns:
        Mots-clés normalisés en minuscules
    """
    if not keywords or not isinstance(keywords, str):
        return keywords
    
    # Sépare, normalise en minuscules, et rejoint
    keywords_list = [kw.strip().lower() for kw in keywords.split(",") if kw.strip()]
    
    # Supprime les doublons tout en préservant l'ordre
    seen = set()
    unique_keywords = []
    for kw in keywords_list:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return ", ".join(unique_keywords) if unique_keywords else None


def is_valid_event(event: dict) -> bool:
    """
    Vérifie si un événement est valide (a un titre et une description).
    
    Args:
        event: Dictionnaire de l'événement
        
    Returns:
        True si l'événement est valide
    """
    title = event.get("title_fr", "")
    description = event.get("description_fr", "")
    
    has_title = title and isinstance(title, str) and len(title.strip()) > 0
    has_description = description and isinstance(description, str) and len(description.strip()) > 0
    
    return has_title and has_description


def is_ile_de_france(event: dict) -> bool:
    """
    Vérifie si un événement est en Ile-de-France.
    
    Args:
        event: Dictionnaire de l'événement
        
    Returns:
        True si l'événement est en Ile-de-France
    """
    department = event.get("location_department", "")
    
    if not department:
        # Si pas de département, on garde (on vérifiera avec la région)
        region = event.get("location_region", "")
        return region == "Île-de-France"
    
    # Normalise d'abord le département
    normalized = normalize_department(department)
    
    return normalized in ILE_DE_FRANCE_DEPARTMENTS


def clean_event(event: dict) -> dict:
    """
    Applique toutes les règles de nettoyage à un événement.
    
    Args:
        event: Dictionnaire de l'événement brut
        
    Returns:
        Dictionnaire de l'événement nettoyé
    """
    cleaned = event.copy()
    
    # 1. Nettoie le HTML dans longdescription_fr
    if "longdescription_fr" in cleaned:
        cleaned["longdescription_fr"] = remove_html_tags(cleaned["longdescription_fr"])
    
    # 2. Normalise le département
    if "location_department" in cleaned:
        cleaned["location_department"] = normalize_department(cleaned["location_department"])
    
    # 3. Normalise la ville
    if "location_city" in cleaned:
        cleaned["location_city"] = normalize_city(cleaned["location_city"])
    
    # 4. Normalise les mots-clés
    if "keywords_fr" in cleaned:
        cleaned["keywords_fr"] = normalize_keywords(cleaned["keywords_fr"])
    
    return cleaned


def load_raw_events(path: Path) -> tuple[list[dict], dict]:
    """
    Charge les événements bruts depuis le fichier JSON.
    
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


def save_cleaned_events(events: list[dict], metadata: dict, path: Path) -> None:
    """
    Sauvegarde les événements nettoyés dans un fichier JSON.
    
    Args:
        events: Liste des événements nettoyés
        metadata: Métadonnées de la collecte
        path: Chemin de sortie
    """
    # Crée le dossier si nécessaire
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ajoute les métadonnées de nettoyage
    cleaning_metadata = {
        "source_metadata": metadata,
        "cleaning": {
            "cleaned_at": datetime.utcnow().isoformat() + "Z",
            "total_events_after_cleaning": len(events),
            "rules_applied": [
                "Suppression des balises HTML de longdescription_fr",
                "Normalisation des départements",
                "Normalisation des villes",
                "Exclusion des événements hors Ile-de-France",
                "Exclusion des événements sans titre ou description",
                "Normalisation des mots-clés (minuscules, déduplication)",
            ]
        }
    }
    
    output_data = {
        "metadata": cleaning_metadata,
        "events": events
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Données sauvegardées dans {path}")


def main():
    """Fonction principale de nettoyage."""
    logger.info("=" * 60)
    logger.info("NETTOYAGE DES DONNEES D'EVENEMENTS")
    logger.info("=" * 60)
    
    # Charge les données brutes
    events, metadata = load_raw_events(INPUT_PATH)
    initial_count = len(events)
    
    # Statistiques de nettoyage
    stats = {
        "initial": initial_count,
        "excluded_no_title_desc": 0,
        "excluded_not_idf": 0,
        "html_cleaned": 0,
        "departments_normalized": 0,
        "cities_normalized": 0,
        "keywords_normalized": 0,
    }
    
    cleaned_events = []
    
    for event in events:
        # Filtre 1: Vérifie si l'événement a un titre et une description
        if not is_valid_event(event):
            stats["excluded_no_title_desc"] += 1
            continue
        
        # Filtre 2: Vérifie si l'événement est en Ile-de-France
        if not is_ile_de_france(event):
            stats["excluded_not_idf"] += 1
            continue
        
        # Applique le nettoyage
        original_longdesc = event.get("longdescription_fr", "")
        original_dept = event.get("location_department", "")
        original_city = event.get("location_city", "")
        original_keywords = event.get("keywords_fr", "")
        
        cleaned = clean_event(event)
        
        # Compte les modifications
        if original_longdesc and cleaned.get("longdescription_fr") != original_longdesc:
            stats["html_cleaned"] += 1
        
        if original_dept and cleaned.get("location_department") != original_dept:
            stats["departments_normalized"] += 1
        
        if original_city and cleaned.get("location_city") != original_city:
            stats["cities_normalized"] += 1
        
        if original_keywords and cleaned.get("keywords_fr") != original_keywords:
            stats["keywords_normalized"] += 1
        
        cleaned_events.append(cleaned)
    
    stats["final"] = len(cleaned_events)
    
    # Affiche les statistiques
    logger.info("-" * 60)
    logger.info("STATISTIQUES DE NETTOYAGE")
    logger.info("-" * 60)
    logger.info(f"Evenements initiaux        : {stats['initial']}")
    logger.info(f"Exclus (sans titre/desc)   : {stats['excluded_no_title_desc']}")
    logger.info(f"Exclus (hors IDF)          : {stats['excluded_not_idf']}")
    logger.info(f"HTML nettoye               : {stats['html_cleaned']}")
    logger.info(f"Departements normalises    : {stats['departments_normalized']}")
    logger.info(f"Villes normalisees         : {stats['cities_normalized']}")
    logger.info(f"Mots-cles normalises       : {stats['keywords_normalized']}")
    logger.info(f"Evenements finaux          : {stats['final']}")
    logger.info("-" * 60)
    
    # Sauvegarde
    save_cleaned_events(cleaned_events, metadata, OUTPUT_PATH)

    logger.info("Nettoyage termine avec succes")

    return 0


if __name__ == "__main__":
    exit(main())