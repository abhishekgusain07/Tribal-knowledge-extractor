FROM python:3.12-slim

LABEL maintainer="abhishekgusain07"
LABEL description="Tribal Knowledge — extract codebase context files for AI tools"

# System deps for git + tree-sitter native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Install the package itself
RUN pip install --no-cache-dir -e .

# Default working directory for mounted repos
WORKDIR /repo

ENTRYPOINT ["tribal-knowledge"]
CMD ["--help"]
