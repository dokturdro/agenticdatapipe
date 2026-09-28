{{
    config(
        materialized='external',
        location=env_var('BIKE_FEAST_HISTORY_PATH'),
        format='parquet'
    )
}}

select *
from {{ ref('int_station_feature_history') }}
