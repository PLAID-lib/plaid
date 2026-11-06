# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE.txt ./
COPY src/plaid ./src/plaid

# Git metadata is deliberately excluded from the build context. Supply a
# release version with --build-arg PLAID_VERSION=... when appropriate.
ARG PLAID_VERSION=0.0.0+docker
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYPLAID=${PLAID_VERSION} \
    uv sync --locked --no-dev --extra viewer --no-editable

FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="PLAID dataset viewer" \
    org.opencontainers.image.source="https://github.com/PLAID-lib/plaid" \
    org.opencontainers.image.licenses="BSD-3-Clause"

# Mesa provides CPU rendering through EGL without an X server or a GPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libglib2.0-0 \
        libhdf5-103-1 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 viewer \
    && mkdir /datasets \
    && chown viewer:viewer /datasets

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    HOME=/home/viewer \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow \
    LIBGL_ALWAYS_SOFTWARE=1

USER viewer
WORKDIR /home/viewer
EXPOSE 8080

ENTRYPOINT ["plaid-viewer", "--host", "0.0.0.0"]
CMD ["--port", "8080", "--datasets-root", "/datasets", "--browse-roots", "/datasets"]