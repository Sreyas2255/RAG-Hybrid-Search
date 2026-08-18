# ============================================================
# FASTAPI HYBRID RAG CONTAINER
# ============================================================

FROM python:3.11-slim

# ============================================================
# ENVIRONMENT
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ============================================================
# APPLICATION SOURCE CODE
# ============================================================

COPY src ./src

# ============================================================
# RAG DATA
# ============================================================

# Source PDF documents
COPY data/raw ./data/raw

# Existing Chroma vector database
COPY data/chroma ./data/chroma

# ============================================================
# FASTAPI PORT
# ============================================================

EXPOSE 8000

# ============================================================
# START FASTAPI
# ============================================================

CMD [ "uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"]