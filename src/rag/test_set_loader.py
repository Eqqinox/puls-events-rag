"""
Utilitaire pour charger et manipuler le jeu de test annoté.

Ce module permet de charger les questions/réponses de référence
et de faciliter l'évaluation du système RAG.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


# Chemin du jeu de test
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_SET_PATH = BASE_DIR / "data" / "evaluation" / "test_set.json"


class TestSetLoader:
    """
    Classe pour charger et manipuler le jeu de test annoté.
    """

    def __init__(self, test_set_path: Path = TEST_SET_PATH):
        """
        Initialise le loader avec le chemin du jeu de test.

        Args:
            test_set_path: Chemin vers le fichier JSON du jeu de test
        """
        self.test_set_path = test_set_path
        self.data = self._load_test_set()

    def _load_test_set(self) -> Dict[str, Any]:
        """
        Charge le jeu de test depuis le fichier JSON.

        Returns:
            Dictionnaire contenant le jeu de test complet

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            json.JSONDecodeError: Si le JSON est invalide
        """
        if not self.test_set_path.exists():
            raise FileNotFoundError(
                f"Fichier de test introuvable: {self.test_set_path}"
            )

        with open(self.test_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def get_all_questions(self) -> List[Dict[str, Any]]:
        """
        Retourne toutes les questions du jeu de test.

        Returns:
            Liste de dictionnaires contenant les questions
        """
        return self.data["questions"]

    def get_question_by_id(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        Récupère une question spécifique par son ID.

        Args:
            question_id: ID de la question

        Returns:
            Dictionnaire de la question ou None si non trouvée
        """
        for question in self.data["questions"]:
            if question["id"] == question_id:
                return question
        return None

    def get_questions_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Filtre les questions par catégorie.

        Args:
            category: Catégorie recherchée (ex: "lieu + thème")

        Returns:
            Liste des questions correspondant à la catégorie
        """
        return [
            q for q in self.data["questions"]
            if q["category"] == category
        ]

    def get_questions_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """
        Filtre les questions par niveau de difficulté.

        Args:
            difficulty: Niveau de difficulté ("easy", "medium", "hard")

        Returns:
            Liste des questions du niveau spécifié
        """
        return [
            q for q in self.data["questions"]
            if q["difficulty"] == difficulty
        ]

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées du jeu de test.

        Returns:
            Dictionnaire des métadonnées
        """
        return self.data["metadata"]

    def get_evaluation_criteria(self) -> Dict[str, str]:
        """
        Retourne les critères d'évaluation.

        Returns:
            Dictionnaire des critères d'évaluation
        """
        return self.data["evaluation_criteria"]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques sur le jeu de test.

        Returns:
            Dictionnaire avec les statistiques
        """
        questions = self.data["questions"]

        # Comptage par catégorie
        categories = {}
        for q in questions:
            cat = q["category"]
            categories[cat] = categories.get(cat, 0) + 1

        # Comptage par difficulté
        difficulties = {}
        for q in questions:
            diff = q["difficulty"]
            difficulties[diff] = difficulties.get(diff, 0) + 1

        return {
            "total_questions": len(questions),
            "categories": categories,
            "difficulties": difficulties,
            "difficulty_distribution": self.data.get("difficulty_distribution", {})
        }

    def print_summary(self):
        """
        Affiche un résumé du jeu de test.
        """
        metadata = self.get_metadata()
        stats = self.get_statistics()

        print("=" * 60)
        print("JEU DE TEST ANNOTÉ - PULS-EVENTS RAG")
        print("=" * 60)
        print(f"Version: {metadata['version']}")
        print(f"Créé le: {metadata['created_at']}")
        print(f"Description: {metadata['description']}")
        print()
        print(f"Nombre total de questions: {stats['total_questions']}")
        print()
        print("Distribution par difficulté:")
        for diff, count in stats['difficulties'].items():
            print(f"  - {diff}: {count}")
        print()
        print("Distribution par catégorie:")
        for cat, count in sorted(stats['categories'].items()):
            print(f"  - {cat}: {count}")
        print("=" * 60)

    def export_questions_only(self, output_path: Optional[Path] = None) -> List[str]:
        """
        Exporte uniquement les questions (sans réponses de référence).

        Utile pour tester le système sans biais.

        Args:
            output_path: Chemin de sortie optionnel pour sauvegarder en JSON

        Returns:
            Liste des questions
        """
        questions = [q["question"] for q in self.data["questions"]]

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)

        return questions


def main():
    """Fonction de test du loader."""
    try:
        # Chargement du jeu de test
        loader = TestSetLoader()

        # Affichage du résumé
        loader.print_summary()

        # Exemple : récupération d'une question
        print("\nExemple de question:")
        question = loader.get_question_by_id(1)
        if question:
            print(f"Q: {question['question']}")
            print(f"Catégorie: {question['category']}")
            print(f"Difficulté: {question['difficulty']}")
            print(f"R: {question['reference_answer']}")

        return 0

    except Exception as e:
        print(f"Erreur: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
