FROM python:3.10-bullseye

# Use Debian Bullseye which is more stable
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install mkdocs mkdocs-material pyyaml -q

COPY . .

EXPOSE 8000

CMD python scripts/update_mkdocs_yaml.py && mkdocs serve -a 0.0.0.0:8000
