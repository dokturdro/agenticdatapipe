{% macro training_source() %}
  {% set source_name = var('training_source') %}
  {% if source_name == 'synthetic' %}
    read_parquet('{{ env_var("BIKE_SYNTHETIC_RAW_HISTORY_PATH") }}')
  {% elif source_name == 'collected' %}
    delta_scan('{{ env_var("BIKE_OBSERVATIONS_URI") }}')
  {% else %}
    {{ exceptions.raise_compiler_error(
      "training_source must be 'synthetic' or 'collected', got: " ~ source_name
    ) }}
  {% endif %}
{% endmacro %}
