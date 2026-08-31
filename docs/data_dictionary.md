# Data dictionary

## Clean_Dataset.csv (modeling)

| Column | Role |
|---|---|
| airline | Categorical input |
| flight | Not used in the production model (`use_flight_code: false`) |
| source_city, destination_city | Categorical + geocoded |
| departure_time, arrival_time | Period buckets |
| stops | zero / one / two_or_more |
| class | Economy / Business |
| duration | Hours |
| days_left | Booking window |
| price | Target |

## Reference airports

`data/reference/airports.csv`: city, airport, iata, state, country, latitude, longitude, aliases.
