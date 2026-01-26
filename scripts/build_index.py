#!/usr/bin/env python3
"""
Script de build complet de l'index vectoriel Faiss.

Ce script orchestre l'ensemble du pipeline de construction de l'index vectoriel :
1. Collecte des événements depuis l'API Open Agenda
2. Nettoyage et validation des données
3. Chunking (découpage en morceaux)
4. Vectorisation (génération des embeddings)
5. Création de l'index Faiss

Usage:
    python scripts/build_index.py

Prérequis:
    - Variable d'environnement MISTRAL_API_KEY configurée
    - Variable d'environnement OPEN_AGENDA_API_KEY configurée
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Ajoute le répertoire racine au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def print_separator(char="=", length=80):
    """Affiche une ligne de séparation."""
    print(char * length)


def print_step_header(step_num: int, total_steps: int, step_name: str):
    """Affiche l'en-tête d'une étape."""
    print_separator()
    print(f"ÉTAPE {step_num}/{total_steps} - {step_name}")
    print_separator()
    print()


def print_step_result(success: bool, message: str, stats: dict = None):
    """Affiche le résultat d'une étape."""
    print()
    if success:
        logger.info(f"✅ {message}")
        if stats:
            for key, value in stats.items():
                logger.info(f"   • {key}: {value}")
    else:
        logger.error(f"❌ {message}")
    print()


def step_1_collect_events() -> bool:
    """
    Étape 1 : Collecte des événements depuis l'API Open Agenda.

    Returns:
        True si succès, False sinon
    """
    try:
        from src.preprocessing.fetch_events import main as fetch_main

        logger.info("Lancement de la collecte des événements...")
        result = fetch_main()

        if result == 0:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Erreur lors de la collecte : {e}", exc_info=True)
        return False


def step_2_clean_events() -> bool:
    """
    Étape 2 : Nettoyage et validation des données.

    Returns:
        True si succès, False sinon
    """
    try:
        from src.preprocessing.clean_events import main as clean_main

        logger.info("Lancement du nettoyage des données...")
        result = clean_main()

        if result == 0:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Erreur lors du nettoyage : {e}", exc_info=True)
        return False


def step_3_chunk_events() -> bool:
    """
    Étape 3 : Découpage des événements en chunks.

    Returns:
        True si succès, False sinon
    """
    try:
        from src.preprocessing.chunk_events import main as chunk_main

        logger.info("Lancement du chunking...")
        result = chunk_main()

        if result == 0:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Erreur lors du chunking : {e}", exc_info=True)
        return False


def step_4_vectorize_events() -> bool:
    """
    Étape 4 : Génération des embeddings (vectorisation).

    Returns:
        True si succès, False sinon
    """
    try:
        from src.preprocessing.vectorize_events import main as vectorize_main

        logger.info("Lancement de la vectorisation...")
        result = vectorize_main()

        if result == 0:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Erreur lors de la vectorisation : {e}", exc_info=True)
        return False


def step_5_build_faiss_index() -> bool:
    """
    Étape 5 : Création de l'index Faiss.

    Returns:
        True si succès, False sinon
    """
    try:
        from src.vectorstore.faiss_index import main as faiss_main

        logger.info("Lancement de la création de l'index Faiss...")
        result = faiss_main()

        if result == 0:
            return True
        else:
            return False

    except Exception as e:
        logger.error(f"Erreur lors de la création de l'index : {e}", exc_info=True)
        return False


def main():
    """Point d'entrée principal du script."""
    start_time = time.time()

    print()
    print_separator("=")
    print("BUILD COMPLET DE L'INDEX VECTORIEL FAISS")
    print_separator("=")
    print()
    logger.info(f"Début du build : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_steps = 5
    steps = [
        {
            "num": 1,
            "name": "Collecte des événements",
            "function": step_1_collect_events,
            "success_msg": "Événements collectés avec succès"
        },
        {
            "num": 2,
            "name": "Nettoyage des données",
            "function": step_2_clean_events,
            "success_msg": "Données nettoyées avec succès"
        },
        {
            "num": 3,
            "name": "Chunking (découpage)",
            "function": step_3_chunk_events,
            "success_msg": "Chunking effectué avec succès"
        },
        {
            "num": 4,
            "name": "Vectorisation (embeddings)",
            "function": step_4_vectorize_events,
            "success_msg": "Vectorisation effectuée avec succès"
        },
        {
            "num": 5,
            "name": "Création de l'index Faiss",
            "function": step_5_build_faiss_index,
            "success_msg": "Index Faiss créé avec succès"
        }
    ]

    # Exécution de chaque étape
    for step in steps:
        print_step_header(step["num"], total_steps, step["name"])

        success = step["function"]()

        if success:
            print_step_result(True, step["success_msg"])
        else:
            print_step_result(False, f"Échec de l'étape {step['num']}")
            logger.error("Le build a échoué. Arrêt du pipeline.")
            return 1

    # Calcul du temps total
    end_time = time.time()
    build_time = end_time - start_time
    minutes = int(build_time // 60)
    seconds = int(build_time % 60)

    # Affichage du résumé final
    print()
    print_separator("=")
    print("BUILD TERMINÉ AVEC SUCCÈS")
    print_separator("=")
    print()
    logger.info(f"✅ Toutes les étapes ont été complétées avec succès")
    logger.info(f"⏱️  Temps total : {minutes} min {seconds} sec")
    logger.info(f"📊 Index vectoriel prêt à l'emploi")
    print()
    print_separator("=")
    print()

    logger.info("Vous pouvez maintenant lancer l'API avec :")
    logger.info("   uvicorn src.api.main:app --reload")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Build interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale : {e}", exc_info=True)
        sys.exit(1)
