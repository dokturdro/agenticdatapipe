select
    cast(station_id as varchar) as station_id,
    to_timestamp(cast(last_reported as bigint)) as event_timestamp,
    cast(num_bikes_available as bigint) as available_bikes,
    cast(num_docks_available as bigint) as available_docks,
    cast(capacity as bigint) as capacity,
    '{{ var("training_source") }}' as source_name
from {{ training_source() }}
