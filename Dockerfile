# Dockerfile pour l'API REST Puls-Events RAG
#
# Build:
#   docker build -t puls-events-api .
#
# Run:
#   docker run -p 8000:8000 -e MISTRAL_API_KEY=xxx puls-events-api

# ==============================================================================
# Etape 1: Image de base
# ==============================================================================

FROM python:3.11-slim

# Métadonnées de l'image
LABEL maintainer="Puls-Events"
LABEL description="API REST pour système RAG d'événements culturels parisiens"
LABEL version="1.0.0"

# Variables d'environnement Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Définit le répertoire de travail
WORKDIR /app

# ==============================================================================
# Etape 2: Installation des dépendances système (si nécessaires)
# ==============================================================================

# Mise à jour des paquets système et installation des dépendances minimales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Etapee 3: Installation des dépendances Python
# ==============================================================================

# Copie uniquement le fichier requirements.txt d'abord
# (pour optimiser le cache Docker)
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Etape 4: Copie du code source
# ==============================================================================

# Copie la structure du projet
COPY src/ ./src/
COPY scripts/ ./scripts/

# ==============================================================================
# Etape 5: Copie de l'index Faiss pré-construit
# ==============================================================================

# IMPORTANT: L'index Faiss doit être construit AVANT de builder l'image Docker
# Exécuter: python scripts/build_index.py
COPY src/data/processed/faiss_index/ ./src/data/processed/faiss_index/

# ==============================================================================
# Etape 6: Configuration de l'application
# ==============================================================================

# Exposition du port de l'API
EXPOSE 8000

# Vérification de la santé du conteneur (optionnel)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ==============================================================================
# Etape 7: Commande de démarrage
# ==============================================================================

# Lance l'API avec uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
