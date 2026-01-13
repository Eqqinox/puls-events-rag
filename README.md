# Puls-Events RAG - Assistant intelligent de recommandation d'evenements culturels

![Python](https://img.shields.io/badge/Python-3.11-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![Status](https://img.shields.io/badge/Status-POC-orange)

## Objectifs du projet

POC (Proof of Concept) d'un systeme RAG (Retrieval-Augmented Generation) pour Puls-Events.

**Mission** : Creer un chatbot intelligent capable de repondre aux questions des utilisateurs sur les evenements culturels a Paris et en Ile-de-France, en s'appuyant sur les donnees de l'API Open Agenda.

**Stack technique** :

- LangChain + Mistral (generation de reponses)
- Faiss (base vectorielle)
- FastAPI (API REST)
- Docker (conteneurisation)

## Structure du projet

```
.
├── src/
│   ├── data/
│   │   ├── raw/                  # Donnees brutes
│   │   └── processed/            # Donnees traitees
│   ├── preprocessing/            # Pipeline de preprocessing
│   ├── vectorstore/              # Indexation Faiss
│   ├── rag/                      # Systeme RAG avec LangChain
│   └── api/                      # API FastAPI
├── tests/                        # Tests unitaires
├── notebooks/                    # Notebooks d'exploration
├── docs/                         # Documentation
├── requirements.txt              # Dependances
└── pyproject.toml                # Configuration du projet
```

## Installation

### Prerequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de packages)
- Cle API Mistral

### Etapes

1. **Cloner le depot**

```bash
git clone https://github.com/<ton-username>/puls-events-rag.git
cd puls-events-rag
```

2. **Creer l'environnement et installer les dependances**

```bash
uv venv
source .venv/bin/activate  # Linux/macOS
uv sync
```

3. **Configurer les variables d'environnement**

```bash
cp .env.example .env
# Editer .env et ajouter votre cle API Mistral
```

4. **Verifier l'installation**

```bash
uv run tests/test_imports.py
```

## Configuration

Creer un fichier `.env` a la racine :

```
MISTRAL_API_KEY=votre_cle_api
```

> Ne jamais commiter le fichier `.env`

---

## Pipeline de preprocessing

Le preprocessing transforme les donnees brutes de l'API Open Agenda en vecteurs prets pour l'indexation Faiss.

### Vue d'ensemble

```
API Open Agenda
      │
      ▼
┌─────────────────┐
│ 1. Collecte     │  fetch_events.py
│    9 923 events │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Nettoyage    │  clean_events.py
│    9 911 events │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Structuration│  prepare_for_vectorization.py
│    9 911 events │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Chunking     │  chunk_events.py
│   14 767 chunks │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Vectorisation│  vectorize_events.py
│   embeddings.npy│
└─────────────────┘
```

### Execution du pipeline

Executer les scripts dans l'ordre suivant :

```bash
# 1. Collecte des donnees depuis l'API Open Agenda
python src/preprocessing/fetch_events.py

# 2. Nettoyage des donnees (HTML, normalisation)
python src/preprocessing/clean_events.py

# 3. Structuration pour le RAG
python src/preprocessing/prepare_for_vectorization.py

# 4. Decoupage en chunks
python src/preprocessing/chunk_events.py

# 5. Vectorisation avec Mistral Embeddings
python src/preprocessing/vectorize_events.py
```

### Description des scripts

| Script | Description | Entree | Sortie |
|--------|-------------|--------|--------|
| `fetch_events.py` | Collecte les evenements depuis l'API OpenDataSoft | API | `events_raw.json` |
| `clean_events.py` | Nettoie le HTML, normalise departements/villes | `events_raw.json` | `events_cleaned.json` |
| `prepare_for_vectorization.py` | Structure les donnees pour l'embedding | `events_cleaned.json` | `events_for_vectorization.json` |
| `chunk_events.py` | Decoupe les textes en chunks | `events_for_vectorization.json` | `events_chunked.json` |
| `vectorize_events.py` | Genere les embeddings Mistral | `events_chunked.json` | `events_vectorized.json` + `embeddings.npy` |

### Fichiers generes

```
src/data/
├── raw/
│   └── events_raw.json              # 9 923 evenements bruts
└── processed/
    ├── exploration_stats.json       # Statistiques d'exploration
    ├── events_cleaned.json          # 9 911 evenements nettoyes
    ├── events_for_vectorization.json# Donnees structurees
    ├── events_chunked.json          # 14 767 chunks
    ├── events_vectorized.json       # Metadonnees des vecteurs
    └── embeddings.npy               # Matrice (14767, 1024) - 58 MB
```

### Parametres de configuration

**Collecte** :
- Zone : Ile-de-France
- Periode : 1er janvier 2025 - 31 decembre 2026

**Chunking** :
- Taille cible : 800 caracteres
- Overlap : 100 caracteres
- Taille min : 100 caracteres

**Vectorisation** :
- Modele : `mistral-embed`
- Dimension : 1024
- Batch size : 50

---

## Tests

Executer les tests unitaires :

```bash
# Tous les tests
pytest tests/test_preprocessing.py -v

# Avec couverture
pytest tests/test_preprocessing.py -v --cov=src/preprocessing

# Un groupe specifique
pytest tests/test_preprocessing.py::TestChunkEvent -v
```

**Resultats attendus** : 85 tests, 100% passed

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_exploration_data.ipynb` | Analyse exploratoire des donnees Open Agenda |

---

## Licence

Projet academique - Formation Expert en Ingenierie et Science des Donnees