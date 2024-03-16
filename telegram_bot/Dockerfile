# # Use a imagem base do FastAPI
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# # Defina o diretório de trabalho no contêiner
# WORKDIR /app

# # Copie o arquivo requirements.txt para o diretório de trabalho no contêiner
# COPY requirements.txt .

# # Atualize pip e instale as dependências especificadas no requirements.txt
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copie o restante dos arquivos do aplicativo para o diretório de trabalho

# COPY . /app/

# CMD ["uvicorn", "fast_api.main:app", "--host", "0.0.0.0", "--port", "8080"]


# Use a imagem base do FastAPI
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o diretório de trabalho no contêiner
COPY requirements.txt .

# Atualize pip e instale as dependências especificadas no requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante dos arquivos do aplicativo para o diretório de trabalho
COPY . /app/
RUN chmod +x /app/start.sh

WORKDIR /app

# Execute o script de inicialização ao iniciar o contêiner
CMD ["sh", "/app/start.sh"]
