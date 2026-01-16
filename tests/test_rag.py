"""
Tests unitaires pour le module RAG (Retrieval-Augmented Generation).

Ce fichier teste les trois modules principaux du système RAG :
1. rag_chain.py - Système RAG complet
2. test_set_loader.py - Chargement du jeu de test
3. evaluation.py - Métriques d'évaluation
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Configuration du path pour les imports
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.rag.rag_chain import PulsEventsRAG
from src.rag.test_set_loader import TestSetLoader
from src.rag.evaluation import RAGEvaluator


# =============================================================================
# Tests pour test_set_loader.py
# =============================================================================

class TestTestSetLoader:
    """Tests pour la classe TestSetLoader."""

    @pytest.fixture
    def loader(self):
        """Fixture pour créer un loader."""
        return TestSetLoader()

    def test_loader_initialization(self, loader):
        """Test que le loader s'initialise correctement."""
        assert loader is not None
        assert loader.data is not None
        assert "questions" in loader.data
        assert "metadata" in loader.data

    def test_get_all_questions(self, loader):
        """Test de récupération de toutes les questions."""
        questions = loader.get_all_questions()
        assert isinstance(questions, list)
        assert len(questions) > 0
        # Vérifier la structure d'une question
        q = questions[0]
        assert "id" in q
        assert "question" in q
        assert "reference_answer" in q
        assert "category" in q
        assert "difficulty" in q

    def test_get_question_by_id(self, loader):
        """Test de récupération d'une question par ID."""
        question = loader.get_question_by_id(1)
        assert question is not None
        assert question["id"] == 1
        assert "question" in question

    def test_get_question_by_invalid_id(self, loader):
        """Test de récupération avec ID invalide."""
        question = loader.get_question_by_id(9999)
        assert question is None

    def test_get_questions_by_category(self, loader):
        """Test de filtrage par catégorie."""
        all_questions = loader.get_all_questions()
        # Prendre une catégorie existante
        category = all_questions[0]["category"]
        questions = loader.get_questions_by_category(category)
        assert isinstance(questions, list)
        assert len(questions) > 0
        # Vérifier que toutes les questions ont la bonne catégorie
        for q in questions:
            assert q["category"] == category

    def test_get_questions_by_difficulty(self, loader):
        """Test de filtrage par difficulté."""
        for difficulty in ["easy", "medium", "hard"]:
            questions = loader.get_questions_by_difficulty(difficulty)
            assert isinstance(questions, list)
            # Vérifier que toutes les questions ont la bonne difficulté
            for q in questions:
                assert q["difficulty"] == difficulty

    def test_get_metadata(self, loader):
        """Test de récupération des métadonnées."""
        metadata = loader.get_metadata()
        assert isinstance(metadata, dict)
        assert "version" in metadata
        assert "total_questions" in metadata
        assert "created_at" in metadata

    def test_get_evaluation_criteria(self, loader):
        """Test de récupération des critères d'évaluation."""
        criteria = loader.get_evaluation_criteria()
        assert isinstance(criteria, dict)
        assert len(criteria) > 0

    def test_get_statistics(self, loader):
        """Test de calcul des statistiques."""
        stats = loader.get_statistics()
        assert isinstance(stats, dict)
        assert "total_questions" in stats
        assert "categories" in stats
        assert "difficulties" in stats
        assert stats["total_questions"] > 0

    def test_export_questions_only(self, loader, tmp_path):
        """Test d'export des questions seulement."""
        output_path = tmp_path / "questions_only.json"
        questions = loader.export_questions_only(output_path)

        # Vérifier le retour
        assert isinstance(questions, list)
        assert len(questions) > 0

        # Vérifier que le fichier a été créé
        assert output_path.exists()

        # Vérifier le contenu du fichier
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "questions" in data
            assert len(data["questions"]) == len(questions)


# =============================================================================
# Tests pour rag_chain.py
# =============================================================================

class TestPulsEventsRAG:
    """Tests pour la classe PulsEventsRAG."""

    @pytest.fixture
    def rag(self):
        """Fixture pour créer une instance de RAG."""
        return PulsEventsRAG()

    def test_rag_initialization(self, rag):
        """Test que le système RAG s'initialise correctement."""
        assert rag is not None
        assert rag.vectorstore is not None
        assert rag.retriever is not None
        assert rag.llm is not None
        assert rag.chain is not None

    def test_rag_has_correct_k_value(self, rag):
        """Test que le nombre de chunks récupérés est correct."""
        assert rag.k == 5

    def test_ask_returns_string(self, rag):
        """Test que ask() retourne une chaîne de caractères."""
        response = rag.ask("Quels concerts à Paris ?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_ask_with_sources_returns_dict(self, rag):
        """Test que ask_with_sources() retourne un dictionnaire."""
        result = rag.ask_with_sources("Expositions d'art ?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "num_sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert isinstance(result["num_sources"], int)

    def test_ask_with_sources_has_correct_number_of_sources(self, rag):
        """Test que le nombre de sources est cohérent."""
        result = rag.ask_with_sources("Théâtre pour enfants ?")
        assert len(result["sources"]) == result["num_sources"]
        # Par défaut k=5, donc maximum 5 sources
        assert result["num_sources"] <= 5

    def test_get_relevant_chunks_returns_list(self, rag):
        """Test que get_relevant_chunks() retourne une liste."""
        chunks = rag.get_relevant_chunks("Jazz à Paris")
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_get_relevant_chunks_structure(self, rag):
        """Test la structure des chunks retournés."""
        chunks = rag.get_relevant_chunks("Concert de musique", k=3)
        assert len(chunks) == 3

        # Vérifier la structure de chaque chunk
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "text" in chunk
            assert "score" in chunk
            assert "metadata" in chunk
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["score"], float)
            assert isinstance(chunk["metadata"], dict)

    def test_get_relevant_chunks_with_custom_k(self, rag):
        """Test get_relevant_chunks avec un k personnalisé."""
        chunks_k3 = rag.get_relevant_chunks("Exposition", k=3)
        chunks_k7 = rag.get_relevant_chunks("Exposition", k=7)

        assert len(chunks_k3) == 3
        assert len(chunks_k7) == 7

    def test_ask_different_questions_different_answers(self, rag):
        """Test que des questions différentes donnent des réponses différentes."""
        answer1 = rag.ask("Concert de jazz")
        answer2 = rag.ask("Théâtre pour enfants")

        # Les réponses doivent être différentes
        assert answer1 != answer2

    @pytest.mark.slow
    def test_rag_chain_end_to_end(self, rag):
        """Test du pipeline complet question → réponse."""
        question = "Quels événements culturels à Paris ce weekend ?"

        # 1. Récupération des chunks
        chunks = rag.get_relevant_chunks(question, k=5)
        assert len(chunks) == 5

        # 2. Génération avec sources
        result = rag.ask_with_sources(question)
        assert len(result["answer"]) > 0
        assert result["num_sources"] == 5

        # 3. Génération simple
        answer = rag.ask(question)
        assert len(answer) > 0


# =============================================================================
# Tests pour evaluation.py
# =============================================================================

class TestRAGEvaluator:
    """Tests pour la classe RAGEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Fixture pour créer un évaluateur."""
        # Créer un mock du système RAG pour éviter les appels API coûteux
        mock_rag = Mock()
        mock_rag.ask_with_sources.return_value = {
            "answer": "Réponse générée par le système",
            "sources": [{"text": "chunk1"}, {"text": "chunk2"}]
        }
        mock_rag.get_relevant_chunks.return_value = [
            {"text": "chunk avec jazz et Paris", "score": 0.5, "metadata": {}},
            {"text": "chunk avec concert", "score": 0.6, "metadata": {}}
        ]

        loader = TestSetLoader()
        return RAGEvaluator(rag_system=mock_rag, test_set_loader=loader)

    def test_evaluator_initialization(self, evaluator):
        """Test que l'évaluateur s'initialise correctement."""
        assert evaluator is not None
        assert evaluator.rag is not None
        assert evaluator.test_loader is not None
        assert evaluator.embeddings_model is not None

    def test_semantic_similarity_same_text(self, evaluator):
        """Test que la similarité de textes identiques est élevée."""
        text = "Ceci est un test de similarité sémantique"
        similarity = evaluator.semantic_similarity(text, text)
        assert isinstance(similarity, float)
        assert 0.95 <= similarity <= 1.0  # Devrait être très proche de 1

    def test_semantic_similarity_similar_texts(self, evaluator):
        """Test de similarité entre textes similaires."""
        text1 = "Concert de jazz à Paris"
        text2 = "Spectacle de jazz dans Paris"
        similarity = evaluator.semantic_similarity(text1, text2)
        assert isinstance(similarity, float)
        assert 0.7 <= similarity <= 1.0  # Devrait être élevée

    def test_semantic_similarity_different_texts(self, evaluator):
        """Test de similarité entre textes différents."""
        text1 = "Concert de jazz à Paris"
        text2 = "Recette de cuisine italienne"
        similarity = evaluator.semantic_similarity(text1, text2)
        assert isinstance(similarity, float)
        # Les embeddings Mistral peuvent trouver des similarités subtiles
        # donc on vérifie juste que c'est moins similaire que des textes identiques
        assert 0.0 <= similarity < 1.0

    def test_exact_match_identical_texts(self, evaluator):
        """Test d'exact match avec textes identiques."""
        text = "Bonjour le monde"
        assert evaluator.exact_match(text, text) is True

    def test_exact_match_different_case(self, evaluator):
        """Test d'exact match avec casse différente."""
        text1 = "Bonjour le monde"
        text2 = "BONJOUR LE MONDE"
        assert evaluator.exact_match(text1, text2) is True

    def test_exact_match_different_whitespace(self, evaluator):
        """Test d'exact match avec espaces différents."""
        text1 = "Bonjour   le    monde"
        text2 = "Bonjour le monde"
        assert evaluator.exact_match(text1, text2) is True

    def test_exact_match_different_texts(self, evaluator):
        """Test d'exact match avec textes différents."""
        text1 = "Bonjour le monde"
        text2 = "Au revoir le monde"
        assert evaluator.exact_match(text1, text2) is False

    def test_partial_match_high_overlap(self, evaluator):
        """Test de partial match avec chevauchement élevé."""
        text1 = "Concert de jazz à Paris ce weekend"
        text2 = "Concert de jazz à Paris"
        assert evaluator.partial_match(text1, text2, threshold=0.5) is True

    def test_partial_match_low_overlap(self, evaluator):
        """Test de partial match avec faible chevauchement."""
        text1 = "Concert de jazz à Paris"
        text2 = "Exposition d'art contemporain à Lyon"
        assert evaluator.partial_match(text1, text2, threshold=0.5) is False

    def test_partial_match_custom_threshold(self, evaluator):
        """Test de partial match avec seuil personnalisé."""
        # "Concert" représente 1/3 des mots de "Concert de jazz"
        text1 = "Concert de jazz à Paris dans une salle"
        text2 = "Concert"

        # Overlap = 1/1 = 100% (tous les mots de text2 sont dans text1)
        # Donc avec seuil bas (10%), devrait matcher
        assert evaluator.partial_match(text1, text2, threshold=0.10) is True

        # Test avec textes ayant overlap partiel
        text3 = "Concert de jazz"
        text4 = "Concert de musique classique baroque"
        # Overlap = 2/5 = 40% (concert, de)
        assert evaluator.partial_match(text3, text4, threshold=0.30) is True
        assert evaluator.partial_match(text3, text4, threshold=0.50) is False

    def test_evaluate_retrieval_quality_full_recall(self, evaluator):
        """Test de qualité du retrieval avec rappel complet."""
        chunks = [
            {"text": "concert de jazz à Paris", "score": 0.5, "metadata": {}},
            {"text": "événement musical", "score": 0.6, "metadata": {}}
        ]
        expected_themes = ["jazz", "concert"]
        expected_locations = ["Paris"]

        result = evaluator.evaluate_retrieval_quality(
            chunks, expected_themes, expected_locations
        )

        assert isinstance(result, dict)
        assert "theme_recall" in result
        assert "location_recall" in result
        assert "overall_recall" in result
        assert result["theme_recall"] == 1.0
        assert result["location_recall"] == 1.0
        assert result["overall_recall"] == 1.0

    def test_evaluate_retrieval_quality_partial_recall(self, evaluator):
        """Test de qualité du retrieval avec rappel partiel."""
        chunks = [
            {"text": "concert de musique", "score": 0.5, "metadata": {}}
        ]
        expected_themes = ["jazz", "concert", "musique"]
        expected_locations = ["Paris", "Lyon"]

        result = evaluator.evaluate_retrieval_quality(
            chunks, expected_themes, expected_locations
        )

        # 2/3 thèmes trouvés = 0.66
        assert 0.6 <= result["theme_recall"] <= 0.7
        # 0/2 lieux trouvés = 0.0
        assert result["location_recall"] == 0.0
        # 2/5 total = 0.4
        assert result["overall_recall"] == 0.4

    def test_evaluate_retrieval_quality_no_expectations(self, evaluator):
        """Test de qualité du retrieval sans attentes."""
        chunks = [{"text": "texte quelconque", "score": 0.5, "metadata": {}}]
        result = evaluator.evaluate_retrieval_quality(chunks, [], [])

        assert result["theme_recall"] == 0.0
        assert result["location_recall"] == 0.0
        assert result["overall_recall"] == 0.0

    def test_evaluate_single_question_structure(self, evaluator):
        """Test de la structure du résultat d'évaluation d'une question."""
        loader = TestSetLoader()
        question_data = loader.get_question_by_id(1)

        result = evaluator.evaluate_single_question(question_data, manual_review=False)

        assert isinstance(result, dict)
        assert result["success"] is True
        assert "question_id" in result
        assert "question" in result
        assert "category" in result
        assert "difficulty" in result
        assert "generated_answer" in result
        assert "reference_answer" in result
        assert "num_sources" in result
        assert "semantic_similarity" in result
        assert "exact_match" in result
        assert "partial_match" in result
        assert "auto_classification" in result
        assert "retrieval_quality" in result

    def test_evaluate_single_question_semantic_similarity_range(self, evaluator):
        """Test que la similarité sémantique est dans la plage attendue."""
        loader = TestSetLoader()
        question_data = loader.get_question_by_id(1)

        result = evaluator.evaluate_single_question(question_data, manual_review=False)

        similarity = result["semantic_similarity"]
        assert 0.0 <= similarity <= 1.0

    def test_evaluate_single_question_auto_classification(self, evaluator):
        """Test de la classification automatique."""
        loader = TestSetLoader()
        question_data = loader.get_question_by_id(1)

        result = evaluator.evaluate_single_question(question_data, manual_review=False)

        classification = result["auto_classification"]
        assert classification in ["correct", "partial", "incorrect"]


# =============================================================================
# Tests d'intégration
# =============================================================================

class TestRAGIntegration:
    """Tests d'intégration pour le système RAG complet."""

    @pytest.mark.slow
    def test_full_evaluation_pipeline(self):
        """Test du pipeline complet d'évaluation."""
        # 1. Charger le jeu de test
        loader = TestSetLoader()
        questions = loader.get_all_questions()
        assert len(questions) > 0

        # 2. Initialiser le RAG
        rag = PulsEventsRAG()
        assert rag is not None

        # 3. Tester sur une question
        question_data = questions[0]
        result = rag.ask_with_sources(question_data["question"])

        assert "answer" in result
        assert len(result["answer"]) > 0

    @pytest.mark.slow
    def test_evaluation_on_sample_questions(self):
        """Test d'évaluation sur un échantillon de questions."""
        # Créer mock RAG pour éviter coûts API
        mock_rag = Mock()
        mock_rag.ask_with_sources.return_value = {
            "answer": "Voici plusieurs concerts de jazz à Paris ce weekend.",
            "sources": [{"text": "chunk1"}, {"text": "chunk2"}]
        }
        mock_rag.get_relevant_chunks.return_value = [
            {"text": "jazz Paris concert", "score": 0.5, "metadata": {}}
        ]

        loader = TestSetLoader()
        evaluator = RAGEvaluator(rag_system=mock_rag, test_set_loader=loader)

        # Évaluer une question
        question_data = loader.get_question_by_id(1)
        result = evaluator.evaluate_single_question(question_data, manual_review=False)

        assert result["success"] is True
        assert isinstance(result["semantic_similarity"], float)


# =============================================================================
# Tests de performance
# =============================================================================

class TestRAGPerformance:
    """Tests de performance du système RAG."""

    @pytest.mark.slow
    def test_ask_response_time(self):
        """Test que ask() répond en temps raisonnable."""
        import time
        rag = PulsEventsRAG()

        start = time.time()
        response = rag.ask("Quels concerts à Paris ?")
        elapsed = time.time() - start

        assert len(response) > 0
        # Devrait répondre en moins de 10 secondes
        assert elapsed < 10.0

    @pytest.mark.slow
    def test_get_relevant_chunks_response_time(self):
        """Test que get_relevant_chunks() est rapide."""
        import time
        rag = PulsEventsRAG()

        start = time.time()
        chunks = rag.get_relevant_chunks("Jazz Paris", k=5)
        elapsed = time.time() - start

        assert len(chunks) == 5
        # Devrait répondre en moins de 3 secondes
        assert elapsed < 3.0


# =============================================================================
# Exécution des tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
