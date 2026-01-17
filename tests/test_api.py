"""
Tests fonctionnels pour l'API REST du système RAG Puls-Events.

Ce module teste tous les endpoints de l'API avec différents scénarios:
- Endpoint /health
- Endpoint /ask (cas valides et invalides)
- Endpoint /rebuild
- Gestion des erreurs
- Tests d'intégration
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ajoute le répertoire racine au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.api.main import app


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """
    Fixture pour le client de test FastAPI.

    Scope module pour réutiliser le même client (et donc le même RAG)
    dans tous les tests, évitant ainsi de recharger l'index à chaque test.
    """
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# TESTS ENDPOINT /health
# ============================================================================

class TestHealthEndpoint:
    """Tests pour l'endpoint /health."""

    def test_health_returns_200(self, client):
        """Test que /health retourne un status code 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_format(self, client):
        """Test que /health retourne le bon format de réponse."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "ok"

    def test_health_response_schema(self, client):
        """Test que /health respecte le schéma HealthResponse."""
        response = client.get("/health")
        data = response.json()

        # Vérifie que toutes les clés attendues sont présentes
        assert set(data.keys()) == {"status"}
        assert isinstance(data["status"], str)


# ============================================================================
# TESTS ENDPOINT /ask - CAS VALIDES
# ============================================================================

class TestAskEndpointValid:
    """Tests pour l'endpoint /ask avec des requêtes valides."""

    def test_ask_valid_question_returns_200(self, client):
        """Test qu'une question valide retourne un status code 200."""
        response = client.post(
            "/ask",
            json={"question": "Quels concerts de jazz à Paris?"}
        )
        assert response.status_code == 200

    def test_ask_returns_correct_format(self, client):
        """Test que /ask retourne le bon format de réponse."""
        response = client.post(
            "/ask",
            json={"question": "Expositions d'art contemporain à Paris"}
        )
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)

    def test_ask_answer_not_empty(self, client):
        """Test que la réponse générée n'est pas vide."""
        response = client.post(
            "/ask",
            json={"question": "Concerts ce weekend à Paris"}
        )
        data = response.json()

        assert len(data["answer"]) > 0
        assert data["answer"].strip() != ""

    def test_ask_sources_have_correct_schema(self, client):
        """Test que les sources respectent le schéma SourceInfo."""
        response = client.post(
            "/ask",
            json={"question": "Théâtre pour enfants"}
        )
        data = response.json()
        sources = data["sources"]

        if len(sources) > 0:
            source = sources[0]
            assert "title" in source
            assert "location" in source
            assert "date" in source
            assert "url" in source
            assert isinstance(source["title"], str)
            assert isinstance(source["location"], str)
            assert isinstance(source["date"], str)
            assert isinstance(source["url"], str)

    def test_ask_with_different_questions(self, client):
        """Test que différentes questions produisent des réponses différentes."""
        response1 = client.post(
            "/ask",
            json={"question": "Concerts de jazz"}
        )
        response2 = client.post(
            "/ask",
            json={"question": "Expositions de peinture"}
        )

        data1 = response1.json()
        data2 = response2.json()

        # Les réponses devraient être différentes pour des questions différentes
        assert data1["answer"] != data2["answer"]


# ============================================================================
# TESTS ENDPOINT /ask - CAS D'ERREUR
# ============================================================================

class TestAskEndpointErrors:
    """Tests pour l'endpoint /ask avec des requêtes invalides."""

    def test_ask_empty_question_returns_400(self, client):
        """Test qu'une question vide retourne une erreur 400."""
        response = client.post(
            "/ask",
            json={"question": ""}
        )
        assert response.status_code == 400

    def test_ask_whitespace_only_question_returns_400(self, client):
        """Test qu'une question contenant uniquement des espaces retourne 400."""
        response = client.post(
            "/ask",
            json={"question": "   "}
        )
        assert response.status_code == 400

    def test_ask_empty_question_returns_error_detail(self, client):
        """Test que l'erreur 400 contient un message explicite."""
        response = client.post(
            "/ask",
            json={"question": ""}
        )
        data = response.json()

        assert "detail" in data
        assert "vide" in data["detail"].lower()

    def test_ask_missing_question_field_returns_422(self, client):
        """Test qu'une requête sans le champ 'question' retourne 422."""
        response = client.post(
            "/ask",
            json={}
        )
        assert response.status_code == 422

    def test_ask_invalid_json_returns_422(self, client):
        """Test qu'un JSON invalide retourne 422."""
        response = client.post(
            "/ask",
            data="not a json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


# ============================================================================
# TESTS ENDPOINT /rebuild
# ============================================================================

class TestRebuildEndpoint:
    """Tests pour l'endpoint /rebuild."""

    def test_rebuild_returns_200(self, client):
        """Test que /rebuild retourne un status code 200."""
        response = client.post("/rebuild")
        assert response.status_code == 200

    def test_rebuild_returns_correct_format(self, client):
        """Test que /rebuild retourne le bon format de réponse."""
        response = client.post("/rebuild")
        data = response.json()

        assert "status" in data
        assert "message" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["message"], str)

    def test_rebuild_success_status(self, client):
        """Test que /rebuild retourne status 'success'."""
        response = client.post("/rebuild")
        data = response.json()

        assert data["status"] == "success"
        assert "succès" in data["message"].lower() or "rechargé" in data["message"].lower()

    def test_rebuild_rag_still_functional_after(self, client):
        """Test que le RAG est toujours fonctionnel après un rebuild."""
        # Rebuild
        rebuild_response = client.post("/rebuild")
        assert rebuild_response.status_code == 200

        # Vérifie que le RAG fonctionne encore
        ask_response = client.post(
            "/ask",
            json={"question": "Test après rebuild"}
        )
        assert ask_response.status_code == 200


# ============================================================================
# TESTS D'INTÉGRATION
# ============================================================================

class TestIntegration:
    """Tests d'intégration pour l'API complète."""

    def test_full_workflow(self, client):
        """Test le workflow complet: health -> ask -> rebuild -> ask."""
        # 1. Vérifier la santé de l'API
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. Poser une question
        ask_response1 = client.post(
            "/ask",
            json={"question": "Concerts à Paris"}
        )
        assert ask_response1.status_code == 200

        # 3. Rebuild
        rebuild_response = client.post("/rebuild")
        assert rebuild_response.status_code == 200

        # 4. Poser une nouvelle question
        ask_response2 = client.post(
            "/ask",
            json={"question": "Expositions à Paris"}
        )
        assert ask_response2.status_code == 200

    def test_multiple_concurrent_asks(self, client):
        """Test que plusieurs requêtes /ask peuvent être traitées."""
        questions = [
            "Concerts de jazz",
            "Expositions d'art",
            "Théâtre contemporain",
            "Festivals de musique",
            "Événements culturels"
        ]

        responses = []
        for question in questions:
            response = client.post(
                "/ask",
                json={"question": question}
            )
            responses.append(response)

        # Vérifie que toutes les requêtes ont réussi
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data

    def test_error_recovery(self, client):
        """Test que l'API se remet d'une erreur et continue de fonctionner."""
        # 1. Provoque une erreur avec une question vide
        error_response = client.post(
            "/ask",
            json={"question": ""}
        )
        assert error_response.status_code == 400

        # 2. Vérifie que l'API fonctionne toujours avec une requête valide
        valid_response = client.post(
            "/ask",
            json={"question": "Concerts à Paris"}
        )
        assert valid_response.status_code == 200


# ============================================================================
# TESTS DE LA DOCUMENTATION SWAGGER
# ============================================================================

class TestSwaggerDocs:
    """Tests pour la documentation Swagger."""

    def test_swagger_ui_accessible(self, client):
        """Test que la documentation Swagger est accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_accessible(self, client):
        """Test que le schéma OpenAPI JSON est accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_openapi_contains_all_endpoints(self, client):
        """Test que le schéma OpenAPI contient tous les endpoints."""
        response = client.get("/openapi.json")
        data = response.json()
        paths = data["paths"]

        assert "/health" in paths
        assert "/ask" in paths
        assert "/rebuild" in paths
