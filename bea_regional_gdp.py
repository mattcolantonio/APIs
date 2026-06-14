# interested in quarterly GDP data for selected states

import beaapi
import os
from dotenv import load_dotenv
load_dotenv()
beakey = os.getenv('BEA_API_KEY')

RGGI_STATES = [
    'Connecticut', 'Delaware', 'Maine', 'Maryland', 
    'Massachusetts', 'New Hampshire', 'New Jersey', 
    'New York', 'Pennsylvania', 'Rhode Island', 
    'Vermont', 'Virginia', 'District of Columbia'
]

years = list(range(2008,2026))
years_str = [str(y) for y in years]


bea_tbl = beaapi.get_data(
    beakey, 
    datasetname='Regional', 
    TableName='SQGDP1', 
    LineCode=1,
    GeoFips='STATE',
    Year="ALL"
    )

bea_tbl = bea_tbl[
    (bea_tbl['GeoName'].isin(RGGI_STATES)) &
    (bea_tbl['TimePeriod'].str_contains(years_str))
    ]

pattern = r'(' + '|'.join(years_str) + r')'

bea_tbl = bea_tbl[
    bea_tbl['GeoName'].isin(RGGI_STATES) &
    bea_tbl['TimePeriod'].str.contains(pattern, regex=True, na=False)
]

