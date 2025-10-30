#Light Python Image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

#Basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

#Working Directory
WORKDIR /app

#Installing dependencies for code execution
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements.txt

#Installinf spaCy model
ARG SPACY_MODEL=en_core_web_sm
RUN python -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('spacy') else 1)" \
    && python -m spacy download ${SPACY_MODEL} || true

#Copying source code
COPY . /app

#Entrypoint: creates entry directory and runs pipeline
RUN printf '#!/usr/bin/env bash\n'                                      >  /usr/local/bin/entrypoint.sh \
 && printf 'set -euo pipefail\n'                                        >> /usr/local/bin/entrypoint.sh \
 && printf 'OUT_DIR=\"${OUTPUT_DIR:-results}\"\n'                       >> /usr/local/bin/entrypoint.sh \
 && printf '# снять возможные кавычки из .env\n'                        >> /usr/local/bin/entrypoint.sh \
 && printf 'OUT_DIR=\"${OUT_DIR%\\\"}\"; OUT_DIR=\"${OUT_DIR#\\\"}\"\n' >> /usr/local/bin/entrypoint.sh \
 && printf 'OUT_DIR=\"${OUT_DIR%\\\'}\"; OUT_DIR=\"${OUT_DIR#\\\'}\"\n' >> /usr/local/bin/entrypoint.sh \
 && printf 'mkdir -p \"/app/${OUT_DIR}\"\n'                             >> /usr/local/bin/entrypoint.sh \
 && printf 'echo \"[entrypoint] Using OUTPUT_DIR=/app/${OUT_DIR}\";\n'  >> /usr/local/bin/entrypoint.sh \
 && printf 'exec python /app/app.py\n'                               >> /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]