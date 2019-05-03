from sklearn import linear_model
import pandas as pd
import numpy as np
import random
import math

### Read In Police Station Lat Lon Data
police_stations = pd.read_csv("crime_data/Police_Stations_-_Map.csv")
locations = police_stations['LOCATION'].str.split(", ", n = 1, expand = True)
locations.columns = ['Latitude','Longitude']
locations['Latitude'] = locations['Latitude'].str.replace('(', '').astype(float)
locations['Longitude'] = locations['Longitude'].str.replace(')', '').astype(float)
locations = locations.values.tolist()
print(locations)

### Read In Low Income Housing Data
low_income = pd.read_csv("crime_data/Affordable_Rental_Housing_Developments.csv")[['Community Area Number']]
low_income['count'] = 1
low_income['Community Area Number'] = low_income['Community Area Number'].astype(str) + '.0_area'
low_income = low_income.groupby('Community Area Number') \
     .agg({'count': 'sum'})
low_income.columns = ['LI_COUNT']

## Read In Burglary Data
data = pd.read_csv("subset_data.csv").reset_index()
data = data.dropna(subset=['Community_Areas', 'Latitude', 'Longitude', 'Date', 'Police_Beats']) \
    .sort_values(by=['Community_Areas', 'Date', 'Primary_Type']) \
    .reset_index()

### Time Related Features ###
# Days Since Last Burglary In Community Zone
data['day_count_since_010101'] = (data['days_since_010101']/np.timedelta64(1,'h'))/24
data['day_delta'] = data['day_count_since_010101'].diff()
data = data[data['day_delta'] > 0]
# Weekday, Working Hour, Non-Winter Months
data['working_hours'] = np.where((data['hour']>=8) & (data['hour']<19), 1, 0)
data['winter'] = np.where((data['month']>=10) | (data['month']<4), 1, 0)
data['work_day'] = np.where((data['day']>=5), 1, 0)

### Geographical Related Features ###
# Dummies for Community Areas
data['Community_Areas'] = data['Community_Areas'].astype(str) + '_area'
com_dummies = pd.get_dummies(data['Community_Areas'])
data = data.join(com_dummies).set_index(['Community_Areas']).drop('1.0_area', axis=1)

# Community Area Stats
data['count'] = 1
community_area_stats = data.groupby('Community_Areas') \
     .agg({'day_delta':'mean', 'count': 'sum'})
community_area_stats.columns = ['CA_AVG_DELTA','CA_COUNT']
data = data.join(community_area_stats).drop('count', axis=1)

# Low Income Housing
data = data.join(low_income)
data['LI_COUNT'] = data['LI_COUNT'].fillna(0)

### Law Enforcement Related Features ###
# Police Count
data['police'] = data['Police_Beats']

# Heversine Distance to police station
def distance(o_lat, o_lon, e_lat, e_lon):
    radius = 6371 # km
    dlat = math.radians(e_lat-o_lat)
    dlon = math.radians(e_lon-o_lon)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(o_lat)) \
        * math.cos(math.radians(e_lat)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

d = {}
i = 1
for l in locations:
    print(i)
    data["dist_{0}".format(i)] = data.apply(lambda row: distance(row['Latitude'], row['Longitude'], l[0], l[1]), axis=1)
    i = i+1
data['min_distance'] = data.loc[:, 'dist_1':'dist_23'].min(axis=1)
data = data.drop(data.loc[:, 'dist_1':'dist_23'], axis=1)

### Export Data
data.to_csv("training_data_burglary.csv", index = False)
