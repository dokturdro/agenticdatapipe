with sequenced as (
    select
        station_id,
        event_timestamp,
        available_bikes,
        available_docks,
        capacity,
        source_name,
        lead(event_timestamp) over station_window as next_event_timestamp,
        lead(available_bikes) over station_window as next_available_bikes
    from {{ ref('stg_training_observations') }}
    window station_window as (
        partition by station_id
        order by event_timestamp
    )
)

select
    station_id,
    event_timestamp,
    cast(epoch(event_timestamp) as bigint) as feature_observed_at,
    cast(extract(hour from event_timestamp) as bigint) as hour_utc,
    cast(extract(isodow from event_timestamp) - 1 as bigint) as day_of_week,
    available_bikes,
    available_docks,
    capacity,
    cast(available_bikes as double) / capacity as availability_ratio,
    case
        when next_event_timestamp = event_timestamp + interval '15 minutes'
            then cast(next_available_bikes as double)
        else null
    end as target_available_bikes_15m,
    source_name
from sequenced
