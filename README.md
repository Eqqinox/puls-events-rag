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
- Docker (conteneurisation)

## Structure du projet

```
.
├── src/
│   ├── data/
│   │   ├── raw/                  # Données brutes
│   │   └── processed/            # Données traitées
│   ├── preprocessing/            # Pipeline de preprocessing
│   ├── vectorstore/              # Indexation Faiss
│   ├── rag/                      # Système RAG avec LangChain
│   └── api/                      # API FastAPI
├── tests/                        # Tests unitaires
├── notebooks/                    # Notebooks d'exploration
├── docs/                         # Documentation
├── requirements.txt              # Dépendances
└── pyproject.toml                # Configuration du projet
```

## Installation

### Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de packages)
- Clé API Mistral

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
MISTRAL_API_KEY=votre_cle_api
```

> Ne jamais commiter le fichier `.env`

---

## Pipeline de preprocessing

Le preprocessing transforme les données brutes de l'API Open Agenda en vecteurs prêts pour l'indexation Faiss.

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

### Exécution du pipeline

Exécuter les scripts dans l'ordre suivant :

```bash
# 1. Collecte des données depuis l'API Open Agenda
python src/preprocessing/fetch_events.py

# 2. Nettoyage des données (HTML, normalisation)
python src/preprocessing/clean_events.py

# 3. Structuration pour le RAG
python src/preprocessing/prepare_for_vectorization.py

# 4. Découpage en chunks
python src/preprocessing/chunk_events.py

# 5. Vectorisation avec Mistral Embeddings
python src/preprocessing/vectorize_events.py
```

### Description des scripts

| Script | Description | Entrée | Sortie |
|--------|-------------|--------|--------|
| `fetch_events.py` | Collecte les événements depuis l'API OpenDataSoft | API | `events_raw.json` |
| `clean_events.py` | Nettoie le HTML, normalise départements/villes | `events_raw.json` | `events_cleaned.json` |
| `prepare_for_vectorization.py` | Structure les données pour l'embedding | `events_cleaned.json` | `events_for_vectorization.json` |
| `chunk_events.py` | Découpe les textes en chunks | `events_for_vectorization.json` | `events_chunked.json` |
| `vectorize_events.py` | Génère les embeddings Mistral | `events_chunked.json` | `events_vectorized.json` + `embeddings.npy` |

### Fichiers générés

```
src/data/
├── raw/
│   └── events_raw.json              # 9 923 événements bruts
└── processed/
    ├── exploration_stats.json       # Statistiques d'exploration
    ├── events_cleaned.json          # 9 911 événements nettoyés
    ├── events_for_vectorization.json# Données structurées
    ├── events_chunked.json          # 14 767 chunks
    ├── events_vectorized.json       # Métadonnées des vecteurs
    └── embeddings.npy               # Matrice (14767, 1024) - 58 MB
```

### Paramètres de configuration

**Collecte** :
- Zone : Île-de-France
- Période : 1er janvier 2025 - 31 décembre 2026

**Chunking** :
- Taille cible : 800 caractères
- Overlap : 100 caractères
- Taille min : 100 caractères

**Vectorisation** :
- Modèle : `mistral-embed`
- Dimension : 1024
- Batch size : 50

---

## Base vectorielle Faiss

L'index Faiss permet la recherche sémantique rapide parmi les 14 767 chunks d'événements.

### Création de l'index

```bash
# Créer l'index Faiss à partir des embeddings
python src/vectorstore/faiss_index.py
```

Ce script :
- Charge les embeddings Mistral (14 767 vecteurs de dimension 1024)
- Crée un index Faiss de type IndexFlatL2 (recherche exacte)
- Associe les métadonnées complètes à chaque vecteur
- Sauvegarde l'index dans `src/data/processed/faiss_index/`

### Caractéristiques de l'index

| Métrique | Valeur |
|----------|--------|
| Nombre de vecteurs | 14 767 |
| Dimension | 1024 |
| Type d'index | IndexFlatL2 (recherche exacte) |
| Taille index | 57.68 MB |
| Taille métadonnées | 18.15 MB |
| Taille totale | 75.83 MB |
| Temps de création | ~1.7 secondes |

### Métadonnées stockées

Pour chaque chunk, l'index conserve :
- `chunk_id` : Identifiant unique du chunk
- `event_id` : Identifiant de l'événement parent
- `title`, `description` : Informations textuelles
- `date_range`, `date_start`, `date_end` : Période de l'événement
- `location_name`, `city`, `department` : Localisation
- `url`, `image` : Liens externes
- `keywords`, `conditions` : Informations complémentaires

### Utilisation de l'index

```python
from src.vectorstore.faiss_index import load_faiss_index, test_search

# Charger l'index
vectorstore = load_faiss_index()

# Effectuer une recherche sémantique
results = vectorstore.similarity_search_with_score(
    "concert de jazz à Paris ce weekend",
    k=5
)

# Afficher les résultats
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Titre: {doc.metadata['title']}")
    print(f"Lieu: {doc.metadata['city']}")
    print(f"Date: {doc.metadata['date_range']}\n")
```

### Performance de recherche

- Recherche unique : < 2 secondes
- Chargement de l'index : < 5 secondes
- Recherches batch : ~6 secondes par requête (incluant appel API Mistral)

---

## Tests

Exécuter les tests unitaires :

```bash
# Tous les tests du preprocessing
pytest tests/test_preprocessing.py -v

# Tous les tests du vectorstore
pytest tests/test_vectorstore.py -v

# Tous les tests du projet
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=src

# Un groupe spécifique
pytest tests/test_preprocessing.py::TestChunkEvent -v
```

### Résultats attendus

| Module | Tests | Statut |
|--------|-------|--------|
| `test_imports.py` | Vérification des imports | OK |
| `test_preprocessing.py` | 85 tests | 100% passed |
| `test_vectorstore.py` | 39 tests | 100% passed |
| **Total** | **124 tests** | **100% passed** |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_exploration_data.ipynb` | Analyse exploratoire des données Open Agenda |

---

## Métriques du projet

- **Événements collectés** : 9 923
- **Événements nettoyés** : 9 911
- **Chunks générés** : 14 767
- **Vecteurs indexés** : 14 767
- **Dimension embeddings** : 1024
- **Lignes de code** : ~3 360
- **Tests unitaires** : 124 (100% passed)
- **Couverture géographique** : Île-de-France (8 départements, 698 villes)
- **Période couverte** : 01/01/2025 - 31/12/2026

---

## Licence

Projet académique - Formation Expert en Ingénierie et Science des Données
