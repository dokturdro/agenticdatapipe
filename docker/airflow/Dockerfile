FROM apache/airflow:slim-3.3.1-python3.12

WORKDIR /opt/airflow/project
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src

# Pin Airflow to the base-image version so project dependency resolution cannot change it.
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    apache-airflow-providers-standard \
    .
