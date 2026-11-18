# The xG API in a container: Python, the API's dependencies, the code and the
# model file, nothing else. No data and no training libraries.
#
#   docker build -t football-ai-api .
#   docker run --rm -p 8000:8000 football-ai-api
#   then open http://127.0.0.1:8000/docs

# Same Python as .python-version, the one the model was trained with
FROM python:3.10-slim

# uv, copied from its official image. Same version as on the laptop
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /app
# Use the Python of the image, never download another one
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, on their own layer: Docker reuses it as long as
# pyproject.toml and uv.lock don't change, so a code change rebuilds in seconds.
# uv's download cache is mounted from outside: it speeds up the next builds
# without ending up inside the image (it weighed 341 MB in there)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project

COPY src ./src
COPY models ./models
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

EXPOSE 8000
# 0.0.0.0: listen to requests coming from outside the container
CMD ["/app/.venv/bin/uvicorn", "football_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]