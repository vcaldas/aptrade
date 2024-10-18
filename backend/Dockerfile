FROM python:3.10-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
# Install the application dependencies.
RUN uv add fastapi[standard] gunicorn
# RUN uv sync --frozen --no-cache
RUN ls -l
# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]
