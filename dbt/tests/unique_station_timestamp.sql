select station_id, event_timestamp
from {{ ref('station_features_history') }}
group by station_id, event_timestamp
having count(*) > 1
