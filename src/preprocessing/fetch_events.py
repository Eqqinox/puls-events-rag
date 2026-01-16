"""
Script de collecte des événements depuis l'API OpenDataSoft.

Ce script récupère les événements culturels d'Île-de-France
depuis l'API OpenDataSoft avec pagination automatique.

Paramètres de collecte :
- Zone : Île-de-France
- Période : 1er janvier 2025 - 31 décembre 2026
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Configuration de l'API
API_BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1"
DATASET_ID = "evenements-publics-openagenda"
RECORDS_PER_PAGE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5  # secondes

# Filtres de collecte
WHERE_CLAUSE = (
    "location_region='Île-de-France' "
    "AND firstdate_begin >= '2025-01-01' "
    "AND firstdate_begin < '2027-01-01'"
)


def fetch_page(offset: int = 0, limit: int = RECORDS_PER_PAGE) -> Optional[dict]:
    """
    Récupère une page de résultats depuis l'API.

    Args:
        offset: Position de départ dans les résultats
        limit: Nombre de résultats à récupérer

    Returns:
        Dictionnaire contenant les résultats ou None en cas d'erreur
    """
    url = f"{API_BASE_URL}/catalog/datasets/{DATASET_ID}/records"
    params = {
        "where": WHERE_CLAUSE,
        "limit": limit,
        "offset": offset,
        "order_by": "firstdate_begin ASC"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning(
                f"Timeout lors de la requête (tentative {attempt}/{MAX_RETRIES})"
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"Erreur HTTP : {e}")
            if response.status_code == 429:  # Too Many Requests
                logger.warning("Rate limit atteint, pause de 60 secondes...")
                time.sleep(60)
            else:
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de requête : {e}")

        if attempt < MAX_RETRIES:
            logger.info(f"Nouvelle tentative dans {RETRY_DELAY} secondes...")
            time.sleep(RETRY_DELAY)

    logger.error(f"Échec après {MAX_RETRIES} tentatives")
    return None


def fetch_all_events() -> list[dict]:
    """
    Récupère tous les événements correspondant aux critères.

    Returns:
        Liste de tous les événements récupérés
    """
    all_events = []
    offset = 0

    # Première requête pour obtenir le total
    logger.info("Démarrage de la collecte des événements...")
    logger.info(f"Filtres appliqués : {WHERE_CLAUSE}")

    first_page = fetch_page(offset=0, limit=1)
    if not first_page:
        logger.error("Impossible de récupérer les informations initiales")
        return []

    total_count = first_page.get("total_count", 0)
    logger.info(f"Nombre total d'événements à récupérer : {total_count}")

    # Collecte paginée
    while offset < total_count:
        page_data = fetch_page(offset=offset)

        if not page_data:
            logger.error(f"Erreur lors de la récupération à l'offset {offset}")
            break

        results = page_data.get("results", [])
        if not results:
            logger.warning(f"Aucun résultat à l'offset {offset}")
            break

        all_events.extend(results)
        offset += RECORDS_PER_PAGE

        # Log de progression
        progress = min(offset, total_count)
        percentage = (progress / total_count) * 100
        logger.info(
            f"Progression : {progress}/{total_count} événements ({percentage:.1f}%)"
        )

        # Pause pour éviter de surcharger l'API
        time.sleep(0.5)

    logger.info(f"Collecte terminée : {len(all_events)} événements récupérés")
    return all_events


def save_events(events: list[dict], output_path: Path) -> bool:
    """
    Sauvegarde les événements dans un fichier JSON.

    Args:
        events: Liste des événements à sauvegarder
        output_path: Chemin du fichier de sortie

    Returns:
        True si la sauvegarde a réussi, False sinon
    """
    try:
        # Création du répertoire parent si nécessaire
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarde avec métadonnées
        output_data = {
            "metadata": {
                "source": "OpenDataSoft API v2.1",
                "dataset": DATASET_ID,
                "filters": WHERE_CLAUSE,
                "total_events": len(events),
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "events": events
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Données sauvegardées dans {output_path}")
        return True

    except IOError as e:
        logger.error(f"Erreur lors de la sauvegarde : {e}")
        return False


def main():
    """Point d'entrée principal du script."""
    # Définition du chemin de sortie
    project_root = Path(__file__).parent.parent #/Users/mounirmeknaci/Desktop/Data_Projects/Projet9/src/
    output_path = project_root / "data" / "raw" / "events_raw.json"

    # Collecte des événements
    events = fetch_all_events()

    if not events:
        logger.error("Aucun événement récupéré, arrêt du script")
        return 1

    # Sauvegarde
    if save_events(events, output_path):
        logger.info("Script terminé avec succès")
        return 0
    else:
        logger.error("Échec de la sauvegarde")
        return 1


if __name__ == "__main__":
    exit(main())