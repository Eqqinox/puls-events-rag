"""
Script de vérification de l'API Mistral avec mistral-small-latest.

Ce script teste :
1. La présence de la clé API Mistral dans .env
2. La connexion avec le SDK Mistral natif
3. La connexion avec LangChain-Mistral

Usage:
    python tests/test_mistral_api.py
"""

import os
from dotenv import load_dotenv


def test_mistral_api():
    """Teste la connexion à l'API Mistral avec mistral-small-latest."""

    print("=" * 60)
    print("TEST DE L'API MISTRAL - mistral-small-latest")
    print("=" * 60)

    # Charger les variables d'environnement
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("\n[ERREUR] MISTRAL_API_KEY non trouvée dans .env")
        print("         Créez un fichier .env avec votre clé API")
        return False

    print("\n[OK] Clé API Mistral trouvée")
    print(f"     Clé: {api_key[:8]}...{api_key[-4:]}")

    # Test avec le SDK Mistral
    print("\n" + "-" * 60)
    print("Test 1/2 - SDK Mistral natif")
    print("-" * 60)
    try:
        from mistralai import Mistral

        client = Mistral(api_key=api_key)

        print("Envoi d'une requête test au modèle mistral-small-latest...")
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "user", "content": "Dis bonjour en une phrase courte."}
            ],
            max_tokens=50
        )

        print("[OK] Connexion API Mistral réussie")
        print(f"     Modèle: mistral-small-latest")
        print(f"     Réponse: {response.choices[0].message.content}")

    except ImportError as e:
        print(f"[ERREUR] Module mistralai non installé: {e}")
        print("         Installez avec: uv add mistralai")
        return False
    except Exception as e:
        print(f"[ERREUR] Test SDK Mistral échoué: {e}")
        return False

    # Test avec LangChain
    print("\n" + "-" * 60)
    print("Test 2/2 - LangChain-Mistral")
    print("-" * 60)
    try:
        from langchain_mistralai import ChatMistralAI

        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=api_key,
            max_tokens=50
        )

        print("Envoi d'une requête test via LangChain...")
        response = llm.invoke("Dis bonjour en une phrase courte.")

        print("[OK] Connexion LangChain-Mistral réussie")
        print(f"     Réponse: {response.content}")

    except ImportError as e:
        print(f"[ERREUR] Module langchain-mistralai non installé: {e}")
        print("         Installez avec: uv add langchain-mistralai")
        return False
    except Exception as e:
        print(f"[ERREUR] Test LangChain échoué: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ SUCCÈS - Tous les tests sont passés!")
    print("=" * 60)
    print("Le modèle mistral-small-latest est prêt pour le RAG.")
    print("Vous pouvez passer à l'implémentation de la sous-tâche 4.2")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_mistral_api()
    exit(0 if success else 1)
