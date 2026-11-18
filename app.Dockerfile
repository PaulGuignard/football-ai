# The Streamlit demo in a container. It only talks to the API, so it gets the
# demo's own libraries and none of the model's.
#
#   docker build -f app.Dockerfile -t football-ai-app .
# Easier: `docker compose up` starts it together with the API.

FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # The project itself isn't installed (that would pull in scikit-learn),
    # so Python finds football_ai.demo through this path instead
    PYTHONPATH=/app/src

COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group app --no-install-project

COPY src ./src
COPY app ./app
# The theme: colours and fonts
COPY .streamlit ./.streamlit

EXPOSE 8501
CMD ["/app/.venv/bin/streamlit", "run", "app/streamlit_app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]