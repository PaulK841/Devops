# Dockerfile

# Utilisez une version de Python stable comme 3.10 ou 3.11
FROM python:3.10-slim

# Définir le répertoire de travail DANS le conteneur
WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier TOUT votre code source
# C'est la seule copie de code dont vous avez besoin
COPY src/ .

# Exposer le port
EXPOSE 8000

# Lancer l'API. Uvicorn trouvera app.py car on a copié le dossier src/
# dans le répertoire de travail /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]