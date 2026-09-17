FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY fixtures ./fixtures
RUN pip install --no-cache-dir .

CMD ["bikepipe", "produce"]
