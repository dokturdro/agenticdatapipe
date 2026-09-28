select *
from {{ ref('station_features_history') }}
where capacity <= 0
   or available_bikes < 0
   or available_docks < 0
   or available_bikes > capacity
   or available_docks > capacity
   or availability_ratio < 0
   or availability_ratio > 1
