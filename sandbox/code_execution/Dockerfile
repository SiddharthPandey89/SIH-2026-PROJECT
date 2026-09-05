FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 sandbox

WORKDIR /workspace
RUN chown sandbox:sandbox /workspace

USER sandbox

ENTRYPOINT ["python"]