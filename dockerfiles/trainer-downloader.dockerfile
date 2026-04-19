FROM python:3.10-slim

WORKDIR /workspace

RUN pip install --no-cache-dir huggingface_hub aiohttp pydantic transformers

COPY scripts/ scripts/

ENV PYTHONPATH=/workspace/scripts

ENTRYPOINT ["python", "scripts/trainer_downloader.py"]