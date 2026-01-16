"""
Tests unitaires pour le module vectorstore.

Ce module teste les fonctionnalités de l'index Faiss:
- Chargement de l'index
- Structure et intégrité des données
- Recherche sémantique
- Pertinence des résultats
- Scores de similarité
- Métadonnées
- Performance

Usage:
    pytest tests/test_vectorstore.py -v
    pytest tests/test_vectorstore.py -v --cov=src/vectorstore
"""

import pytest
import time
import numpy as np
from pathlib import Path

# Import des modules à tester
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vectorstore.faiss_index import (
    load_faiss_index,
    load_embeddings_and_metadata,
    build_and_save_faiss_index,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    INDEX_PATH,
)


# Constante pour le nombre de chunks attendu (mis à jour après chunking LangChain)
EXPECTED_CHUNK_COUNT = 15928


# Fixtures pytest
@pytest.fixture(scope="module")
def vectorstore():
    """Fixture pour charger l'index une seule fois pour tous les tests."""
    return load_faiss_index()


@pytest.fixture
def sample_queries():
    """Requêtes de test prédéfinies."""
    return [
        "concert de jazz",
        "exposition d'art",
        "théâtre pour enfants",
        "événement gratuit à Paris",
        "musée",
        "festival"
    ]


# =============================================================================
# TESTS DE CHARGEMENT
# =============================================================================

class TestLoadIndex:
    """Tests de chargement de l'index Faiss."""

    def test_index_files_exist(self):
        """Vérifie que les fichiers de l'index existent."""
        index_file = INDEX_PATH / "index.faiss"
        pkl_file = INDEX_PATH / "index.pkl"

        assert index_file.exists(), f"Fichier index.faiss introuvable: {index_file}"
        assert pkl_file.exists(), f"Fichier index.pkl introuvable: {pkl_file}"

    def test_load_index_success(self):
        """Teste le chargement réussi de l'index."""
        vectorstore = load_faiss_index()
        assert vectorstore is not None, "L'index n'a pas été chargé"
        assert hasattr(vectorstore, 'index'), "L'index n'a pas d'attribut 'index'"

    def test_load_index_with_invalid_path(self):
        """Teste le chargement avec un chemin invalide."""
        invalid_path = Path("/chemin/invalide/inexistant")

        with pytest.raises(FileNotFoundError):
            load_faiss_index(index_path=invalid_path)

    def test_index_not_empty(self, vectorstore):
        """Vérifie que l'index n'est pas vide."""
        assert vectorstore.index.ntotal > 0, "L'index est vide"


# =============================================================================
# TESTS DE STRUCTURE
# =============================================================================

class TestIndexStructure:
    """Tests de la structure de l'index."""

    def test_vector_count(self, vectorstore):
        """Vérifie le nombre de vecteurs dans l'index."""
        actual_count = vectorstore.index.ntotal

        assert actual_count == EXPECTED_CHUNK_COUNT, \
            f"Nombre de vecteurs incorrect: {actual_count}, attendu: {EXPECTED_CHUNK_COUNT}"

    def test_embedding_dimension(self, vectorstore):
        """Vérifie la dimension des embeddings."""
        expected_dim = 1024
        actual_dim = vectorstore.index.d

        assert actual_dim == expected_dim, \
            f"Dimension incorrecte: {actual_dim}, attendu: {expected_dim}"

    def test_index_type(self, vectorstore):
        """Vérifie le type d'index Faiss."""
        index_type = type(vectorstore.index).__name__
        assert "IndexFlat" in index_type, \
            f"Type d'index incorrect: {index_type}"

    def test_metadata_fields(self, vectorstore):
        """Vérifie la présence des champs de métadonnées."""
        results = vectorstore.similarity_search("test", k=1)

        assert len(results) > 0, "Aucun résultat pour tester les métadonnées"

        doc = results[0]
        required_fields = [
            "chunk_id", "event_id", "chunk_index", "total_chunks",
            "uid", "title", "description"
        ]

        for field in required_fields:
            assert field in doc.metadata, f"Champ manquant: {field}"

    def test_metadata_completeness(self, vectorstore):
        """Vérifie que tous les documents ont des métadonnées."""
        sample_size = min(100, vectorstore.index.ntotal)
        results = vectorstore.similarity_search("événement", k=sample_size)

        for doc in results:
            assert doc.metadata is not None, "Métadonnées manquantes"
            assert len(doc.metadata) > 0, "Métadonnées vides"


# =============================================================================
# TESTS DE RECHERCHE BASIQUE
# =============================================================================

class TestSimilaritySearch:
    """Tests de recherche par similarité."""

    def test_search_returns_results(self, vectorstore):
        """Vérifie qu'une recherche retourne des résultats."""
        results = vectorstore.similarity_search("concert", k=5)
        assert len(results) > 0, "Aucun résultat retourné"

    def test_search_k_parameter(self, vectorstore):
        """Vérifie que le paramètre k est respecté."""
        for k in [1, 3, 5, 10]:
            results = vectorstore.similarity_search("exposition", k=k)
            assert len(results) == k, f"Attendu {k} résultats, obtenu {len(results)}"

    def test_search_returns_documents(self, vectorstore):
        """Vérifie que les résultats sont des Documents."""
        results = vectorstore.similarity_search("théâtre", k=3)

        for doc in results:
            assert hasattr(doc, 'page_content'), "Pas d'attribut page_content"
            assert hasattr(doc, 'metadata'), "Pas d'attribut metadata"
            assert isinstance(doc.page_content, str), "page_content n'est pas une string"

    def test_search_with_empty_query(self, vectorstore):
        """Teste le comportement avec une requête vide."""
        # Une requête vide devrait quand même retourner des résultats
        results = vectorstore.similarity_search("", k=5)
        assert len(results) == 5, "Devrait retourner des résultats même avec requête vide"

    def test_search_with_long_query(self, vectorstore):
        """Teste avec une requête longue."""
        long_query = " ".join(["événement culturel"] * 50)  # ~500 caractères
        results = vectorstore.similarity_search(long_query, k=5)
        assert len(results) == 5, "Devrait gérer les requêtes longues"


# =============================================================================
# TESTS DE PERTINENCE
# =============================================================================

class TestSearchRelevance:
    """Tests de pertinence des résultats."""

    def test_jazz_query_relevance(self, vectorstore):
        """Vérifie la pertinence pour une recherche jazz."""
        results = vectorstore.similarity_search("concert de jazz", k=5)

        texts = [doc.page_content.lower() for doc in results]
        keywords = ["jazz", "concert", "musique"]

        relevant_count = sum(
            any(kw in text for kw in keywords)
            for text in texts
        )

        assert relevant_count >= len(results) * 0.4, \
            f"Seulement {relevant_count}/{len(results)} résultats pertinents"

    def test_theater_query_relevance(self, vectorstore):
        """Vérifie la pertinence pour une recherche théâtre."""
        results = vectorstore.similarity_search("théâtre", k=5)

        texts = [doc.page_content.lower() for doc in results]
        keywords = ["théâtre", "spectacle", "pièce", "scène"]

        relevant_count = sum(
            any(kw in text for kw in keywords)
            for text in texts
        )

        assert relevant_count >= len(results) * 0.4, \
            f"Seulement {relevant_count}/{len(results)} résultats pertinents"

    def test_exhibition_query_relevance(self, vectorstore):
        """Vérifie la pertinence pour une recherche exposition."""
        results = vectorstore.similarity_search("exposition", k=5)

        texts = [doc.page_content.lower() for doc in results]
        keywords = ["exposition", "expo", "art", "musée", "galerie"]

        relevant_count = sum(
            any(kw in text for kw in keywords)
            for text in texts
        )

        assert relevant_count >= len(results) * 0.4, \
            f"Seulement {relevant_count}/{len(results)} résultats pertinents"

    def test_paris_query_relevance(self, vectorstore):
        """Vérifie la pertinence pour une recherche Paris."""
        results = vectorstore.similarity_search("événement à Paris", k=10)

        # Vérifie que Paris apparaît dans les métadonnées ou le texte
        paris_count = sum(
            "paris" in doc.metadata.get("city", "").lower() or
            "paris" in doc.page_content.lower()
            for doc in results
        )

        assert paris_count >= len(results) * 0.3, \
            f"Seulement {paris_count}/{len(results)} résultats avec Paris"

    def test_children_query_relevance(self, vectorstore):
        """Vérifie la pertinence pour une recherche enfants."""
        results = vectorstore.similarity_search("activité pour enfants", k=5)

        texts = [doc.page_content.lower() for doc in results]
        keywords = ["enfant", "jeune", "famille", "petit"]

        relevant_count = sum(
            any(kw in text for kw in keywords)
            for text in texts
        )

        assert relevant_count >= len(results) * 0.3, \
            f"Seulement {relevant_count}/{len(results)} résultats pertinents"

    def test_multiple_queries_return_different_results(self, vectorstore, sample_queries):
        """Vérifie que différentes requêtes retournent des résultats différents."""
        all_results = []

        for query in sample_queries[:3]:
            results = vectorstore.similarity_search(query, k=3)
            top_titles = [doc.metadata.get("title", "") for doc in results]
            all_results.append(set(top_titles))

        # Au moins quelques différences entre les résultats
        assert len(all_results[0]) > 0, "Pas de résultats"
        # Les 3 ensembles ne doivent pas être identiques
        assert not (all_results[0] == all_results[1] == all_results[2]), \
            "Toutes les requêtes retournent les mêmes résultats"


# =============================================================================
# TESTS AVEC SCORES
# =============================================================================

class TestSearchWithScore:
    """Tests de recherche avec scores de similarité."""

    def test_search_returns_scores(self, vectorstore):
        """Vérifie que les scores sont retournés."""
        results = vectorstore.similarity_search_with_score("concert", k=5)

        assert len(results) == 5, "Devrait retourner 5 résultats"

        for doc, score in results:
            assert isinstance(score, (int, float, np.floating)), "Score n'est pas numérique"

    def test_scores_are_ordered(self, vectorstore):
        """Vérifie que les scores sont ordonnés (croissant = meilleur)."""
        results = vectorstore.similarity_search_with_score("exposition", k=10)

        scores = [score for _, score in results]

        # Les scores doivent être en ordre croissant (distance L2)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], \
                f"Scores non ordonnés: {scores[i]} > {scores[i + 1]}"

    def test_score_range(self, vectorstore):
        """Vérifie que les scores sont dans une plage raisonnable."""
        results = vectorstore.similarity_search_with_score("théâtre", k=10)

        scores = [score for _, score in results]

        # Pour des embeddings normalisés, scores devraient être entre 0 et 2
        for score in scores:
            assert 0.0 <= score <= 2.0, f"Score hors plage: {score}"

    def test_identical_query_similar_scores(self, vectorstore):
        """Vérifie que la même requête donne des scores similaires.
        
        Note: L'API Mistral peut avoir de légères variations entre les appels,
        donc on utilise une tolérance plus large.
        """
        query = "concert de musique classique"

        results1 = vectorstore.similarity_search_with_score(query, k=5)
        results2 = vectorstore.similarity_search_with_score(query, k=5)

        scores1 = [score for _, score in results1]
        scores2 = [score for _, score in results2]

        # Les scores doivent être similaires (tolérance pour variations API Mistral)
        for s1, s2 in zip(scores1, scores2):
            assert abs(s1 - s2) < 1e-3, f"Scores trop différents: {s1} vs {s2}"

    def test_top_result_has_best_score(self, vectorstore):
        """Vérifie que le premier résultat a le meilleur score."""
        results = vectorstore.similarity_search_with_score("festival", k=10)

        scores = [score for _, score in results]

        # Le premier score doit être le plus petit (meilleur)
        assert scores[0] == min(scores), "Le premier résultat n'a pas le meilleur score"


# =============================================================================
# TESTS DES MÉTADONNÉES
# =============================================================================

class TestMetadataRetrieval:
    """Tests de récupération des métadonnées."""

    def test_metadata_fields_present(self, vectorstore):
        """Vérifie que tous les champs requis sont présents."""
        results = vectorstore.similarity_search("événement", k=10)

        required_fields = [
            "chunk_id", "event_id", "chunk_index", "total_chunks",
            "uid", "title", "description"
        ]

        for doc in results:
            for field in required_fields:
                assert field in doc.metadata, \
                    f"Champ {field} manquant dans {doc.metadata.get('chunk_id', 'unknown')}"

    def test_metadata_types(self, vectorstore):
        """Vérifie les types des métadonnées."""
        results = vectorstore.similarity_search("concert", k=5)

        for doc in results:
            meta = doc.metadata

            # Vérifications de types
            assert isinstance(meta.get("chunk_id"), str), "chunk_id doit être string"
            assert isinstance(meta.get("event_id"), str), "event_id doit être string"
            assert isinstance(meta.get("chunk_index"), int), "chunk_index doit être int"
            assert isinstance(meta.get("total_chunks"), int), "total_chunks doit être int"
            assert isinstance(meta.get("title"), str), "title doit être string"

    def test_event_id_format(self, vectorstore):
        """Vérifie le format des event_id."""
        results = vectorstore.similarity_search("théâtre", k=10)

        for doc in results:
            event_id = doc.metadata.get("event_id")
            assert event_id, "event_id manquant"
            assert len(event_id) > 0, "event_id vide"

    def test_url_format(self, vectorstore):
        """Vérifie le format des URLs."""
        results = vectorstore.similarity_search("exposition", k=10)

        for doc in results:
            url = doc.metadata.get("url")
            if url:  # URL peut être vide pour certains événements
                assert isinstance(url, str), "URL doit être string"
                # Vérifie que c'est une URL valide (basique)
                assert url.startswith("http") or url == "", \
                    f"URL invalide: {url}"

    def test_chunk_index_consistency(self, vectorstore):
        """Vérifie la cohérence des index de chunks."""
        results = vectorstore.similarity_search("festival", k=20)

        for doc in results:
            chunk_index = doc.metadata.get("chunk_index")
            total_chunks = doc.metadata.get("total_chunks")

            assert 0 <= chunk_index < total_chunks, \
                f"Index incohérent: {chunk_index}/{total_chunks}"

    def test_location_fields(self, vectorstore):
        """Vérifie les champs de localisation."""
        results = vectorstore.similarity_search("Paris", k=10)

        for doc in results:
            # Au moins un champ de localisation doit être présent
            has_location = (
                doc.metadata.get("city") or
                doc.metadata.get("department") or
                doc.metadata.get("location_name")
            )
            assert has_location, "Aucun champ de localisation"


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

class TestIntegration:
    """Tests du pipeline complet."""

    def test_full_search_pipeline(self, vectorstore):
        """Teste le pipeline complet de recherche."""
        # Recherche
        query = "concert de jazz à Paris"
        results = vectorstore.similarity_search_with_score(query, k=5)

        # Vérifications
        assert len(results) == 5, "Devrait retourner 5 résultats"

        for doc, score in results:
            # Vérifications de structure
            assert doc.page_content, "Texte vide"
            assert doc.metadata, "Métadonnées vides"
            assert isinstance(score, (float, np.floating)), "Score invalide"

            # Vérifications de métadonnées
            assert doc.metadata.get("title"), "Titre manquant"
            assert doc.metadata.get("event_id"), "event_id manquant"

    def test_load_embeddings_and_metadata(self):
        """Teste le chargement des embeddings et métadonnées."""
        embeddings, chunks = load_embeddings_and_metadata()

        assert embeddings.shape[0] == len(chunks), \
            "Nombre d'embeddings et chunks incohérent"
        assert embeddings.shape[1] == 1024, "Dimension incorrecte"
        assert len(chunks) == EXPECTED_CHUNK_COUNT, \
            f"Nombre de chunks incorrect: {len(chunks)}, attendu: {EXPECTED_CHUNK_COUNT}"

    def test_index_persistence(self):
        """Vérifie que l'index peut être rechargé."""
        # Charge l'index deux fois
        vs1 = load_faiss_index()
        vs2 = load_faiss_index()

        # Vérifie qu'ils ont les mêmes propriétés
        assert vs1.index.ntotal == vs2.index.ntotal, \
            "Nombre de vecteurs différent après rechargement"

        # Vérifie qu'ils retournent les mêmes résultats
        query = "exposition d'art"
        results1 = vs1.similarity_search(query, k=3)
        results2 = vs2.similarity_search(query, k=3)

        titles1 = [doc.metadata.get("title") for doc in results1]
        titles2 = [doc.metadata.get("title") for doc in results2]

        assert titles1 == titles2, "Résultats différents après rechargement"

    def test_multiple_searches(self, vectorstore, sample_queries):
        """Teste plusieurs recherches consécutives."""
        for query in sample_queries:
            results = vectorstore.similarity_search(query, k=3)
            assert len(results) == 3, f"Échec pour la requête: {query}"
            assert all(doc.page_content for doc in results), \
                f"Résultats vides pour: {query}"

    def test_different_k_values(self, vectorstore):
        """Teste différentes valeurs de k."""
        query = "concert"

        for k in [1, 5, 10, 20, 50]:
            results = vectorstore.similarity_search(query, k=k)
            assert len(results) == k, f"Échec pour k={k}"


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

class TestPerformance:
    """Tests de performance de la recherche."""

    def test_search_speed(self, vectorstore):
        """Vérifie que la recherche est rapide."""
        query = "concert de musique"

        start = time.time()
        vectorstore.similarity_search(query, k=10)
        duration = time.time() - start

        # La recherche devrait prendre moins de 2 secondes
        # (incluant l'appel API Mistral pour l'embedding de la requête)
        assert duration < 2.0, f"Recherche trop lente: {duration:.2f}s"

    def test_load_index_speed(self):
        """Vérifie que le chargement de l'index est rapide."""
        start = time.time()
        load_faiss_index()
        duration = time.time() - start

        # Le chargement devrait prendre moins de 3 secondes
        assert duration < 3.0, f"Chargement trop lent: {duration:.2f}s"

    def test_batch_search(self, vectorstore, sample_queries):
        """Teste les performances sur un batch de recherches."""
        start = time.time()

        for query in sample_queries:
            vectorstore.similarity_search(query, k=5)

        duration = time.time() - start

        # 6 recherches en moins de 40 secondes (incluant appels API Mistral)
        assert duration < 40.0, \
            f"Batch search trop lent: {duration:.2f}s pour {len(sample_queries)} requêtes"

        avg_time = duration / len(sample_queries)
        assert avg_time < 7.0, \
            f"Temps moyen par recherche trop élevé: {avg_time:.2f}s"