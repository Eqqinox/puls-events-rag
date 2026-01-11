# Puls-Events RAG - Assistant intelligent de recommandation d'événements culturels

![Python](https://img.shields.io/badge/Python-3.11-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![Status](https://img.shields.io/badge/Status-POC-orange)

## Objectifs du projet

POC (Proof of Concept) d'un système RAG (Retrieval-Augmented Generation) pour Puls-Events.

**Mission** : Créer un chatbot intelligent capable de répondre aux questions des utilisateurs sur les événements culturels à Paris et en Île-de-France, en s'appuyant sur les données de l'API Open Agenda.

**Stack technique** :
- LangChain + Mistral (génération de réponses)
- Faiss (base vectorielle)
- FastAPI (API REST)

## Structure du projet
```
.
├── src/
│   ├── preprocessing/    # Collecte et nettoyage des données Open Agenda
│   ├── vectorstore/      # Indexation Faiss
│   ├── rag/              # Système RAG avec LangChain
│   └── api/              # API FastAPI
├── tests/                # Tests unitaires
├── data/                 # Données (non versionné)
├── notebooks/            # Notebooks d'exploration
├── docs/                 # Documentation
├── requirements.txt      # Dépendances (généré depuis uv.lock)
└── pyproject.toml        # Configuration du projet
```

## Installation

### Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de packages)

### Étapes

1. **Cloner le dépôt**
```bash
   git clone https://github.com/<ton-username>/puls-events-rag.git
   cd puls-events-rag
```

2. **Créer l'environnement et installer les dépendances**
```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   uv sync
```

3. **Configurer les variables d'environnement**
```bash
   cp .env.example .env
   # Éditer .env et ajouter votre clé API Mistral
```

4. **Vérifier l'installation**
```bash
   uv run tests/test_imports.py
```

## Configuration

Créer un fichier `.env` à la racine :
```
MISTRAL_API_KEY=votre_clé_api
```

> Ne jamais commiter le fichier `.env`

## Licence

Projet académique - Formation Expert en Ingénierie et Science des Données