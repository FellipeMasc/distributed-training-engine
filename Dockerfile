FROM python:3.11-slim

RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace

COPY scripts/ /workspace/scripts/
