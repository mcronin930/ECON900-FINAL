from sklearn import linear_model
import pandas as pd
import random

# The data to load
f = "crime_data/Crimes_-_2001_to_present.csv"

# Code to load random sample of .csv
num_lines = sum(1 for l in open(f))
print(num_lines)
# Sample size - in this case ~20% - Anymore than this I run into memory issues
size = int(num_lines / 5)
skip_idx = random.sample(range(1, num_lines), num_lines - size)

# Read the data
## According to Chicago Data Portal, 'Community Areas' is current.
## I am going to remove 'Community Area'.
drops = ['Case Number', 'Block',  'Description',  'Location Description',
                    'Updated On', 'Community Area']
data = pd.read_csv(f, skiprows=skip_idx).drop(drops, axis=1)
data.columns = ['ID','Date','IUCR',
                     'Primary_Type','Arrest','Domestic', 'Beat', 'District',
                     'Ward', 'FBI_Code', 'X_Coordinate', 'Y_Coordinate',
                     'Year', 'Latitude', 'Longitude', 'Location',
                     'Historical_Wards', 'Zip Codes', 'Community_Areas',
                     'Census_Tracts', 'Wards', 'Boundaries_ZIP',
                     'Police_Dist', 'Police_Beats']


data['Date'] = pd.to_datetime(data.Date)
data['hour'] = data.Date.dt.hour
data['day'] = data.Date.dt.weekday
data['month'] = data.Date.dt.month
data['jan012001'] = "1/01/2001  0:00:00 AM"
data['jan012001'] = data['jan012001'].astype('datetime64[ns]')
data['days_since_010101'] = data['Date'] - data['jan012001']


BURGLARY = data[data.Primary_Type == 'BURGLARY']

print(BURGLARY.dtypes)
print(BURGLARY.head())
BURGLARY.to_csv("subset_data.csv", index = False)
print(BURGLARY['hour'].value_counts())
print(BURGLARY['day'].value_counts())
print(BURGLARY['month'].value_counts())
print(BURGLARY['Community_Areas'].value_counts())
print(BURGLARY['Ward'].value_counts())
print(data['Primary_Type'].value_counts())
