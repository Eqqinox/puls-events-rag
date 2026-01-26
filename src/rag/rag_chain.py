"""
Module RAG (Retrieval-Augmented Generation) pour Puls-Events.

Ce module orchestre la recherche sémantique (Faiss) et la génération
de réponses (Mistral) pour créer un chatbot de recommandations d'événements.

Usage:
    from src.rag.rag_chain import PulsEventsRAG

    rag = PulsEventsRAG()
    response = rag.ask("Quels concerts de jazz à Paris ce weekend ?")
    print(response)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Ajoute le répertoire racine au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.vectorstore.faiss_index import load_faiss_index

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Charge les variables d'environnement
load_dotenv()

# Chemin de l'index (BASE_DIR déjà défini en haut)
INDEX_PATH = BASE_DIR / "src" / "data" / "processed" / "faiss_index"
EVENTS_PATH = BASE_DIR / "src" / "data" / "processed" / "events_cleaned.json"


def get_database_stats() -> Dict[str, Any]:
    """
    Calcule les statistiques de la base d'événements.

    Returns:
        Dictionnaire avec les statistiques de la base
    """
    import json

    try:
        with open(EVENTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        events = data.get('events', [])
        total_events = len(events)

        # Compter les événements par année
        count_2025 = 0
        count_2026 = 0

        for event in events:
            first_begin = event.get('firstdate_begin', '')
            first_end = event.get('firstdate_end', '')
            last_begin = event.get('lastdate_begin', '')
            last_end = event.get('lastdate_end', '')

            all_dates = f'{first_begin} {first_end} {last_begin} {last_end}'

            if '2026' in all_dates:
                count_2026 += 1
            if '2025' in all_dates:
                count_2025 += 1

        return {
            'total_events': total_events,
            'events_2025': count_2025,
            'events_2026': count_2026,
            'pct_2025': round(count_2025 / total_events * 100, 1) if total_events > 0 else 0,
            'pct_2026': round(count_2026 / total_events * 100, 1) if total_events > 0 else 0
        }

    except Exception as e:
        logger.warning(f"Impossible de charger les statistiques de la base: {e}")
        # Retourne des stats par défaut en cas d'erreur
        return {
            'total_events': 0,
            'events_2025': 0,
            'events_2026': 0,
            'pct_2025': 0,
            'pct_2026': 0
        }


# Template de prompt pour les recommandations d'événements culturels
def get_system_prompt(db_stats: Optional[Dict[str, Any]] = None):
    """Génère le prompt système avec la date actuelle et les statistiques de la base."""
    from datetime import datetime

    now = datetime.now()
    current_date = now.strftime("%d/%m/%Y")
    current_year = now.year
    current_month = now.strftime("%B %Y")
    current_day_name = now.strftime("%A")

    # Charger les stats si non fournies
    if db_stats is None:
        db_stats = get_database_stats()

    # Construire la section statistiques
    stats_section = f"""
STATISTIQUES DE LA BASE :
- Total d'événements en base : {db_stats['total_events']:,} événements
- Événements 2025 : {db_stats['events_2025']:,} ({db_stats['pct_2025']}%)
- Événements 2026 : {db_stats['events_2026']:,} ({db_stats['pct_2026']}%)
- Période couverte : 2025-2026
- Zone géographique : Paris et Île-de-France"""

    return f"""Tu es un assistant intelligent spécialisé dans les recommandations d'événements culturels à Paris et en Île-de-France.

Ta mission est d'aider les utilisateurs à découvrir des événements culturels pertinents (concerts, expositions, théâtre, spectacles, etc.) en fonction de leurs questions.

INFORMATIONS CONTEXTUELLES :
- Date actuelle : {current_date} ({current_day_name})
- Année en cours : {current_year}
- Mois en cours : {current_month}
- Zone géographique couverte : Paris et Île-de-France uniquement
{stats_section}

CONTEXTE DES ÉVÉNEMENTS :
{{context}}

RÈGLES À RESPECTER :
1. Base-toi UNIQUEMENT sur les événements fournis dans le contexte ci-dessus
2. Réponds en français de manière claire et concise
3. Si plusieurs événements correspondent, présente les 3-5 plus pertinents
4. Pour chaque événement recommandé, inclus :
   - Le titre
   - Le lieu (nom du lieu + ville)
   - La date ou période
   - Une brève description
5. Si aucun événement ne correspond, dis-le clairement et propose des alternatives similaires
6. Sois naturel et conversationnel dans tes réponses
7. Ne jamais inventer d'informations qui ne sont pas dans le contexte
8. Si l'utilisateur pose une question qui n'est pas en rapport avec la recherche d'événements culturels, réponds poliment que tu es conçu uniquement pour aider à trouver des événements culturels et que tu ne peux malheureusement pas répondre à d'autres types de questions
9. Si l'utilisateur demande des événements en dehors de Paris et d'Île-de-France, indique que tu ne disposes pas d'informations sur les événements en dehors de cette zone géographique
10. Utilise la date actuelle pour interpréter correctement les références temporelles comme "ce weekend", "cette semaine", "ce mois-ci"
11. Pour les questions de comptage ("combien", "nombre de"), utilise les statistiques de la base fournies ci-dessus plutôt que de compter les événements dans le contexte
12. Si, dans la question, l'année n'est pas spécifiée, tu répondras avec les événements de l'année en cours seulement

QUESTION DE L'UTILISATEUR :
{{question}}

RÉPONSE :"""

# Variable globale pour compatibilité avec les tests
SYSTEM_PROMPT = get_system_prompt()


class PulsEventsRAG:
    """
    Système RAG pour les recommandations d'événements culturels.

    Cette classe orchestre :
    - Le chargement de l'index Faiss
    - La recherche sémantique (retrieval)
    - La génération de réponses avec Mistral (generation)
    """

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        model_name: str = "mistral-small-latest",
        k: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 500,
        api_key: Optional[str] = None
    ):
        """
        Initialise le système RAG.

        Args:
            index_path: Chemin vers l'index Faiss
            model_name: Nom du modèle Mistral à utiliser
            k: Nombre de chunks à récupérer pour le contexte
            temperature: Température pour la génération (0-1, plus bas = plus déterministe)
            max_tokens: Nombre maximum de tokens dans la réponse
            api_key: Clé API Mistral (optionnel, utilise .env par défaut)

        Raises:
            ValueError: Si la clé API Mistral n'est pas trouvée
            FileNotFoundError: Si l'index Faiss n'existe pas
        """
        logger.info("Initialisation du système RAG Puls-Events...")

        # Configuration de la clé API
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key
        elif not os.getenv("MISTRAL_API_KEY"):
            raise ValueError(
                "MISTRAL_API_KEY non trouvée. "
                "Définissez-la dans .env ou passez-la en paramètre."
            )

        self.index_path = index_path
        self.model_name = model_name
        self.k = k
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Chargement de l'index Faiss
        logger.info(f"Chargement de l'index Faiss depuis {index_path}...")
        self.vectorstore = load_faiss_index(index_path)
        logger.info("Index Faiss chargé avec succès")

        # Configuration du retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        logger.info(f"Retriever configuré (k={k})")

        # Configuration du LLM
        logger.info(f"Configuration du modèle {model_name}...")
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("MISTRAL_API_KEY")
        )
        logger.info("Modèle configuré avec succès")

        # Chargement des statistiques de la base
        logger.info("Chargement des statistiques de la base...")
        self.db_stats = get_database_stats()
        logger.info(f"Statistiques chargées: {self.db_stats['total_events']} événements "
                   f"({self.db_stats['events_2025']} en 2025, {self.db_stats['events_2026']} en 2026)")

        # Configuration du prompt (régénéré avec date actuelle et stats à chaque initialisation)
        self.prompt = ChatPromptTemplate.from_template(get_system_prompt(self.db_stats))

        # Construction de la chaîne RAG
        self.chain = self._build_chain()

        logger.info("Système RAG prêt")

    def _build_chain(self):
        """
        Construit la chaîne RAG complète.

        Returns:
            Chaîne LangChain orchestrant retrieval + génération
        """
        # Fonction pour formatter les documents récupérés
        def format_docs(docs) -> str:
            """Formate les documents pour le contexte du prompt."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                text = doc.page_content
                metadata = doc.metadata

                # Formate chaque événement avec ses métadonnées
                event_info = f"[Événement {i}]\n"
                event_info += f"Titre: {metadata.get('title', 'N/A')}\n"
                event_info += f"Description: {text}\n"
                event_info += f"Lieu: {metadata.get('location_name', 'N/A')}, {metadata.get('city', 'N/A')}\n"
                event_info += f"Date: {metadata.get('date_range', 'N/A')}\n"

                if metadata.get('url'):
                    event_info += f"URL: {metadata.get('url')}\n"

                event_info += "\n"
                formatted.append(event_info)

            return "\n".join(formatted)

        # Construction de la chaîne avec LCEL (LangChain Expression Language)
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def ask(self, question: str) -> str:
        """
        Pose une question au système RAG et obtient une réponse.

        Args:
            question: Question de l'utilisateur en langage naturel

        Returns:
            Réponse générée par le système RAG

        Raises:
            ValueError: Si la question est vide
            Exception: En cas d'erreur lors de la génération

        Example:
            >>> rag = PulsEventsRAG()
            >>> response = rag.ask("Quels concerts de jazz à Paris ?")
            >>> print(response)
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide")

        logger.info(f"Question: {question}")

        try:
            # Invocation de la chaîne RAG
            response = self.chain.invoke(question)

            logger.info("Réponse générée avec succès")
            return response

        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            raise

    def ask_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Pose une question et retourne la réponse avec les sources.

        Args:
            question: Question de l'utilisateur

        Returns:
            Dictionnaire contenant la réponse et les chunks sources

        Example:
            >>> result = rag.ask_with_sources("Concerts à Paris ?")
            >>> print(result["answer"])
            >>> for source in result["sources"]:
            >>>     print(source.metadata["title"])
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide")

        logger.info(f"Question (avec sources): {question}")

        try:
            # Récupération des documents sources
            sources = self.retriever.invoke(question)

            # Génération de la réponse
            answer = self.ask(question)

            return {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }

        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise

    def get_relevant_chunks(self, question: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Récupère les chunks pertinents sans générer de réponse.

        Utile pour le debugging ou l'évaluation du retrieval.

        Args:
            question: Question de l'utilisateur
            k: Nombre de chunks à récupérer (par défaut: self.k)

        Returns:
            Liste des chunks avec métadonnées et scores
        """
        k = k or self.k

        # Recherche avec scores
        results = self.vectorstore.similarity_search_with_score(question, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            })

        return formatted_results


def main():
    """Fonction de test du module RAG."""
    try:
        logger.info("=" * 60)
        logger.info("TEST DU SYSTÈME RAG PULS-EVENTS")
        logger.info("=" * 60)

        # Initialisation du RAG
        rag = PulsEventsRAG()

        # Questions de test
        test_questions = [
            "Quels concerts de jazz à Paris ce weekend ?",
            "Y a-t-il des expositions d'art contemporain ?",
            "Recommande-moi du théâtre pour enfants"
        ]

        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- Question {i}/3 ---")
            logger.info(f"Q: {question}")

            try:
                response = rag.ask(question)
                logger.info(f"R: {response}\n")
            except Exception as e:
                logger.error(f"Erreur: {e}")

        logger.info("=" * 60)
        logger.info("Test terminé avec succès")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Erreur lors du test: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
