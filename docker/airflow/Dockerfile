FROM apache/airflow:slim-3.3.1-python3.12

WORKDIR /opt/airflow/project
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src
COPY --chown=airflow:root dbt ./dbt

# keep the base image Airflow version
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    apache-airflow-providers-standard \
    ".[dbt]"

# preinstall so dbt runs offline
RUN python -c "import duckdb; c=duckdb.connect(); c.execute('INSTALL delta'); c.execute('INSTALL httpfs')"
