datasets
-----------
Different datasets wrapper functions that could be used for different purposes. Details are in the docstring of each function.

### Example

This is an example to show how to call the county lookup table. 

```python
from skills_ml.datasets.sba_city_county import county_lookup

lookup = county_lookup()

# This will give you a dictionary of cities in Illinois
print lookup['IL'] 

# This will tell you what county and its fips county code 
# the city belongs to
print lookup['IL']['Chicago'] 

```