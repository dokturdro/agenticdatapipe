FROM apache/airflow:slim-3.3.1-python3.12

WORKDIR /opt/airflow/project
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src ./src
COPY --chown=airflow:root dbt ./dbt

# Pin Airflow to the base-image version so project dependency resolution cannot change it.
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    apache-airflow-providers-standard \
    ".[dbt]"

# Cache the extensions in the image so dbt does not need internet access at runtime.
RUN python -c "import duckdb; c=duckdb.connect(); c.execute('INSTALL delta'); c.execute('INSTALL httpfs')"
