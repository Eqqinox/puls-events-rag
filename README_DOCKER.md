# Guide Docker - Puls-Events RAG API

Ce guide explique comment conteneuriser et déployer l'API REST Puls-Events avec Docker.

---

## Prérequis

- Docker installé (version 20.10 ou supérieure)
- Index Faiss pré-construit dans `src/data/processed/faiss_index/`
- Clé API Mistral

---

## Construction de l'image Docker

### Étape 1 : Vérifier que l'index Faiss existe

Avant de builder l'image Docker, vous devez avoir construit l'index Faiss :

```bash
# Vérifier la présence de l'index
ls -lh src/data/processed/faiss_index/

# Si l'index n'existe pas, le construire
python scripts/build_index.py
```

### Étape 2 : Builder l'image Docker

```bash
# Build de l'image (peut prendre quelques minutes)
docker build -t puls-events-api .

# Vérifier que l'image a été créée
docker images | grep puls-events-api
```

**Taille estimée de l'image** : ~1.2 GB (Python 3.11-slim + dépendances + index Faiss)

---

## Lancement du conteneur

### Lancement simple

```bash
docker run -p 8000:8000 \
  -e MISTRAL_API_KEY=your_mistral_api_key \
  puls-events-api
```

### Lancement en mode détaché (background)

```bash
docker run -d \
  --name puls-events-api \
  -p 8000:8000 \
  -e MISTRAL_API_KEY=your_mistral_api_key \
  puls-events-api
```

### Lancement avec fichier .env

```bash
docker run -d \
  --name puls-events-api \
  -p 8000:8000 \
  --env-file .env \
  puls-events-api
```

---

## Tests de l'API conteneurisée

Une fois le conteneur lancé, tester les endpoints :

```bash
# Health check
curl http://localhost:8000/health

# Documentation Swagger
open http://localhost:8000/docs

# Test du endpoint /ask
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts de jazz à Paris ce weekend?"}'
```

---

## Gestion du conteneur

### Voir les logs

```bash
# Logs en temps réel
docker logs -f puls-events-api

# Dernières 100 lignes
docker logs --tail 100 puls-events-api
```

### Arrêter le conteneur

```bash
docker stop puls-events-api
```

### Redémarrer le conteneur

```bash
docker restart puls-events-api
```

### Supprimer le conteneur

```bash
# Arrêter et supprimer
docker stop puls-events-api
docker rm puls-events-api
```

### Supprimer l'image

```bash
docker rmi puls-events-api
```

---

## Debugging

### Accéder au shell du conteneur

```bash
docker exec -it puls-events-api /bin/bash
```

### Vérifier les variables d'environnement

```bash
docker exec puls-events-api env
```

### Vérifier la structure des fichiers

```bash
docker exec puls-events-api ls -la /app/src/data/processed/faiss_index/
```

---

## Fichiers Docker

| Fichier | Description |
|---------|-------------|
| `Dockerfile` | Définition de l'image Docker |
| `.dockerignore` | Fichiers à exclure du build |
| `README_DOCKER.md` | Ce guide |

---

## Ressources

- [Documentation Docker](https://docs.docker.com/)
- [Documentation FastAPI avec Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Best practices Dockerfile](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
