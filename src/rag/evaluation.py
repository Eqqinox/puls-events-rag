"""
Module d'évaluation pour le système RAG Puls-Events.

Ce module implémente différentes métriques pour évaluer la qualité
des réponses générées par le système RAG.

Métriques disponibles :
- Similarité sémantique (via embeddings)
- Exact match
- Classification manuelle (correct / partiellement correct / incorrect)
- Métriques de retrieval (précision, rappel)

Usage:
    python src/rag/evaluation.py
Sortie:
    src/data/evaluation/evaluation_results.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Ajoute le répertoire racine au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_mistralai import MistralAIEmbeddings
import numpy as np

from src.rag.rag_chain import PulsEventsRAG
from src.rag.test_set_loader import TestSetLoader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Charge les variables d'environnement
load_dotenv()


class RAGEvaluator:
    """
    Classe d'évaluation du système RAG.

    Implémente diverses métriques pour évaluer la qualité
    des réponses générées.
    """

    def __init__(
        self,
        rag_system: Optional[PulsEventsRAG] = None,
        test_set_loader: Optional[TestSetLoader] = None
    ):
        """
        Initialise l'évaluateur.

        Args:
            rag_system: Système RAG à évaluer (optionnel, créé si None)
            test_set_loader: Loader du jeu de test (optionnel, créé si None)
        """
        logger.info("Initialisation de l'évaluateur RAG...")

        # Initialise le système RAG
        if rag_system is None:
            logger.info("Création du système RAG...")
            self.rag = PulsEventsRAG()
        else:
            self.rag = rag_system

        # Initialise le loader de jeu de test
        if test_set_loader is None:
            logger.info("Chargement du jeu de test...")
            self.test_loader = TestSetLoader()
        else:
            self.test_loader = test_set_loader

        # Initialise le modèle d'embeddings pour la similarité sémantique
        logger.info("Initialisation du modèle d'embeddings...")
        self.embeddings_model = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY")
        )

        logger.info("Évaluateur prêt")

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité sémantique entre deux textes.

        Utilise les embeddings Mistral et la similarité cosinus.

        Args:
            text1: Premier texte
            text2: Deuxième texte

        Returns:
            Score de similarité entre 0 et 1 (1 = identique)
        """
        # Génère les embeddings
        emb1 = np.array(self.embeddings_model.embed_query(text1))
        emb2 = np.array(self.embeddings_model.embed_query(text2))

        # Calcule la similarité cosinus
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        similarity = dot_product / (norm1 * norm2)

        # Normalise entre 0 et 1
        similarity = (similarity + 1) / 2

        return float(similarity)

    def exact_match(self, generated: str, reference: str) -> bool:
        """
        Vérifie si deux textes sont identiques (exact match).

        Args:
            generated: Texte généré
            reference: Texte de référence

        Returns:
            True si identiques, False sinon
        """
        # Normalise (minuscules, espaces)
        gen_normalized = " ".join(generated.lower().split())
        ref_normalized = " ".join(reference.lower().split())

        return gen_normalized == ref_normalized

    def partial_match(self, generated: str, reference: str, threshold: float = 0.5) -> bool:
        """
        Vérifie si le texte généré contient une partie significative de la référence.

        Args:
            generated: Texte généré
            reference: Texte de référence
            threshold: Seuil de correspondance (0-1)

        Returns:
            True si correspondance partielle, False sinon
        """
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())

        if len(ref_words) == 0:
            return False

        overlap = len(gen_words.intersection(ref_words))
        overlap_ratio = overlap / len(ref_words)

        return overlap_ratio >= threshold

    def manual_classification(
        self,
        question: str,
        generated: str,
        reference: str,
        auto_suggestion: Optional[str] = None
    ) -> str:
        """
        Support pour la classification manuelle des réponses.

        Affiche la question, la réponse générée et la référence,
        puis demande à l'utilisateur de classifier la réponse.

        Args:
            question: Question posée
            generated: Réponse générée
            reference: Réponse de référence
            auto_suggestion: Suggestion automatique basée sur la similarité

        Returns:
            Classification : "correct", "partial", ou "incorrect"
        """
        print("\n" + "=" * 80)
        print("CLASSIFICATION MANUELLE")
        print("=" * 80)
        print(f"\nQuestion : {question}")
        print(f"\nRéponse générée :\n{generated}")
        print(f"\nRéponse de référence :\n{reference}")

        if auto_suggestion:
            print(f"\nSuggestion automatique : {auto_suggestion}")

        print("\nOptions de classification :")
        print("  1. correct    - La réponse est correcte et complète")
        print("  2. partial    - La réponse est partiellement correcte")
        print("  3. incorrect  - La réponse est incorrecte")

        while True:
            choice = input("\nVotre classification (1/2/3) : ").strip()
            if choice == "1":
                return "correct"
            elif choice == "2":
                return "partial"
            elif choice == "3":
                return "incorrect"
            else:
                print("Choix invalide. Veuillez entrer 1, 2 ou 3.")

    def evaluate_retrieval_quality(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        expected_themes: List[str],
        expected_locations: List[str]
    ) -> Dict[str, float]:
        """
        Évalue la qualité du retrieval.

        Vérifie si les chunks récupérés contiennent les thèmes
        et lieux attendus.

        Args:
            retrieved_chunks: Chunks récupérés par le retriever
            expected_themes: Thèmes attendus
            expected_locations: Lieux attendus

        Returns:
            Dictionnaire avec précision et rappel
        """
        # Extraction du texte des chunks
        retrieved_text = " ".join([
            chunk["text"].lower() for chunk in retrieved_chunks
        ])

        # Vérification des thèmes
        themes_found = sum(
            1 for theme in expected_themes
            if theme.lower() in retrieved_text
        )
        theme_recall = themes_found / len(expected_themes) if expected_themes else 0

        # Vérification des lieux
        locations_found = sum(
            1 for loc in expected_locations
            if loc.lower() in retrieved_text
        )
        location_recall = locations_found / len(expected_locations) if expected_locations else 0

        # Calcul du rappel global
        total_expected = len(expected_themes) + len(expected_locations)
        total_found = themes_found + locations_found
        overall_recall = total_found / total_expected if total_expected > 0 else 0

        return {
            "theme_recall": theme_recall,
            "location_recall": location_recall,
            "overall_recall": overall_recall,
            "themes_found": themes_found,
            "locations_found": locations_found
        }

    def evaluate_single_question(
        self,
        question_data: Dict[str, Any],
        manual_review: bool = False
    ) -> Dict[str, Any]:
        """
        Évalue le système sur une seule question.

        Args:
            question_data: Données de la question (du jeu de test)
            manual_review: Si True, demande une classification manuelle

        Returns:
            Résultats de l'évaluation
        """
        question = question_data["question"]
        reference = question_data["reference_answer"]

        logger.info(f"Évaluation de la question {question_data['id']}: {question}")

        # Génération de la réponse
        try:
            result = self.rag.ask_with_sources(question)
            generated = result["answer"]
            sources = result["sources"]
        except Exception as e:
            logger.error(f"Erreur lors de la génération : {e}")
            return {
                "question_id": question_data["id"],
                "error": str(e),
                "success": False
            }

        # Calcul de la similarité sémantique
        similarity_score = self.semantic_similarity(generated, reference)

        # Calcul exact match
        is_exact_match = self.exact_match(generated, reference)

        # Calcul partial match
        is_partial_match = self.partial_match(generated, reference)

        # Évaluation du retrieval
        retrieved_chunks = self.rag.get_relevant_chunks(question)
        retrieval_quality = self.evaluate_retrieval_quality(
            retrieved_chunks,
            question_data.get("expected_themes", []),
            question_data.get("expected_locations", [])
        )

        # Suggestion automatique de classification
        if similarity_score >= 0.8:
            auto_classification = "correct"
        elif similarity_score >= 0.5:
            auto_classification = "partial"
        else:
            auto_classification = "incorrect"

        # Classification manuelle si demandée
        if manual_review:
            manual_classification = self.manual_classification(
                question,
                generated,
                reference,
                auto_suggestion=auto_classification
            )
        else:
            manual_classification = None

        return {
            "question_id": question_data["id"],
            "question": question,
            "category": question_data["category"],
            "difficulty": question_data["difficulty"],
            "generated_answer": generated,
            "reference_answer": reference,
            "num_sources": len(sources),
            "semantic_similarity": similarity_score,
            "exact_match": is_exact_match,
            "partial_match": is_partial_match,
            "auto_classification": auto_classification,
            "manual_classification": manual_classification,
            "retrieval_quality": retrieval_quality,
            "success": True
        }

    def evaluate_all(
        self,
        manual_review: bool = False,
        output_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Évalue le système sur tout le jeu de test.

        Args:
            manual_review: Si True, demande classification manuelle pour chaque question
            output_file: Chemin pour sauvegarder les résultats (optionnel)

        Returns:
            Résultats complets de l'évaluation
        """
        logger.info("=" * 60)
        logger.info("ÉVALUATION COMPLÈTE DU SYSTÈME RAG")
        logger.info("=" * 60)

        questions = self.test_loader.get_all_questions()
        results = []

        for i, question_data in enumerate(questions, 1):
            logger.info(f"\nQuestion {i}/{len(questions)}")
            result = self.evaluate_single_question(question_data, manual_review)
            results.append(result)

        # Calcul des statistiques globales
        successful_results = [r for r in results if r.get("success", False)]

        avg_similarity = np.mean([
            r["semantic_similarity"] for r in successful_results
        ])

        exact_matches = sum(1 for r in successful_results if r["exact_match"])
        partial_matches = sum(1 for r in successful_results if r["partial_match"])

        avg_retrieval_recall = np.mean([
            r["retrieval_quality"]["overall_recall"] for r in successful_results
        ])

        # Classification automatique
        auto_correct = sum(
            1 for r in successful_results
            if r["auto_classification"] == "correct"
        )
        auto_partial = sum(
            1 for r in successful_results
            if r["auto_classification"] == "partial"
        )
        auto_incorrect = sum(
            1 for r in successful_results
            if r["auto_classification"] == "incorrect"
        )

        summary = {
            "total_questions": len(questions),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(questions) - len(successful_results),
            "metrics": {
                "average_semantic_similarity": float(avg_similarity),
                "exact_matches": exact_matches,
                "partial_matches": partial_matches,
                "average_retrieval_recall": float(avg_retrieval_recall)
            },
            "auto_classification": {
                "correct": auto_correct,
                "partial": auto_partial,
                "incorrect": auto_incorrect
            },
            "results": results,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Sauvegarde si demandé
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"\nRésultats sauvegardés dans {output_file}")

        # Affichage du résumé
        self._print_summary(summary)

        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """Affiche un résumé des résultats d'évaluation."""
        logger.info("\n" + "=" * 60)
        logger.info("RÉSUMÉ DE L'ÉVALUATION")
        logger.info("=" * 60)

        metrics = summary["metrics"]
        auto_class = summary["auto_classification"]

        logger.info(f"\nQuestions évaluées : {summary['successful_evaluations']}/{summary['total_questions']}")
        logger.info(f"\nMétriques :")
        logger.info(f"  - Similarité sémantique moyenne : {metrics['average_semantic_similarity']:.3f}")
        logger.info(f"  - Exact matches : {metrics['exact_matches']}")
        logger.info(f"  - Partial matches : {metrics['partial_matches']}")
        logger.info(f"  - Rappel retrieval moyen : {metrics['average_retrieval_recall']:.3f}")
        logger.info(f"\nClassification automatique :")
        logger.info(f"  - Correct : {auto_class['correct']}")
        logger.info(f"  - Partiel : {auto_class['partial']}")
        logger.info(f"  - Incorrect : {auto_class['incorrect']}")
        logger.info("=" * 60)


def main():
    """Fonction principale pour l'évaluation."""
    try:
        # Initialisation de l'évaluateur
        evaluator = RAGEvaluator()

        # Évaluation complète (sans revue manuelle)
        output_path = BASE_DIR / "src" / "data" / "evaluation" / "evaluation_results.json"
        results = evaluator.evaluate_all(
            manual_review=False,
            output_file=output_path
        )

        return 0

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation : {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
