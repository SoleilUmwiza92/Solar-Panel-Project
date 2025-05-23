## Coded (Python) by Peter van Alem, lectorate Future Automotive
import knmy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import time
import urllib
import urllib3
import warnings
from matplotlib.collections import PolyCollection
from pvlib.solarposition import get_solarposition
from pvlib import irradiance

print('\014')
plt.close('all')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Convenience dict
number_to_month = { 
    1 : ['January', 31],
    2 : ['February', 28],
    3 : ['March', 31],
    4 : ['April', 30],
    5 : ['May', 31],
    6 : ['June', 30],
    7 : ['July', 31],
    8 : ['August', 31],
    9 : ['September', 30],
   10 : ['October', 31],
   11 : ['November', 30],
   12 : ['December', 31]
}

class Color:
    W       = '\033[0m'
    R       = '\033[1;31;48m'
    G       = '\033[32m'
    B       = '\033[34m'
    O       = '\033[33m'
    LB      = '\033[1;34;48m'
    P       = '\033[35m'
    C       = '\033[36m'
    WBLACK  = '\033[1;37;40m'
    WBLUE   = '\033[1;37;44m'
    END     = '\033[1;37;0m'
    WR      = '\033[1;37;41m'
    BG      = '\033[1;40;42m'

# Load KNMI data, combine YYYYMMDD and HH
start_date_UTC  = 2024010100
end_date_UTC    = 2024013123
visualisation_year = [2024]
visualisation_month = [1]
number_of_weather_station = 330
print(f'start date UTC = {str(start_date_UTC)}')
print(f'end date UTC = {str(end_date_UTC)}')

url = 'https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens'
# Dutch: Uurvak u loopt van u - 1 UT tot u UT.
try:
    urllib.request.urlretrieve(url)
    disclaimer, stations, variables, df = knmy.knmy.get_hourly_data(
        stations = [number_of_weather_station], start = start_date_UTC, end = end_date_UTC,
        inseason = False, variables = ['WIND', 'TEMP', 'Q', 'SQ', 'VICL', 'P'], parse = True)
    # When KNMI is updating its data, the above statement will produce an empty dataframe.
    # In this case it's preferable to load the dataframe from the former successfull run.
    if df['YYYYMMDD'].isnull().values.any():
        print(Color.R + 'loading KNMI data from file ...' + Color.END)
        df = pd.read_csv('KNMI/df_KNMI.csv')
    else:
        station_name = stations['name'].values[0]
        print(f'Loading data from weather station: {station_name}')
        print('Saving KNMI data to file ...')
       # df.to_csv('~/Documents/Soleil/Lectoraar-Internship/Solar-Panel-Project/Energy/data_experimenting/df_KNMI.csv')
except urllib.error.HTTPError as err:
    print(f'{url}: {err}')
    print(Color.R + 'loading KNMI data from file ...' + Color.END)
    df = pd.read_csv('~/Documents/Soleil/Lectoraar-Internship/Solar-Panel-Project/Energy/data_experimenting/df_KNMI.csv')

df['HH'] = df['HH'].astype(str).str.zfill(2)
df['YYYYMMDDHH'] = df['YYYYMMDD'].astype(str) + df['HH'].astype(str).replace('24','00')
df['YYYYMMDDHH'] = pd.to_datetime(df['YYYYMMDDHH'], format = '%Y%m%d%H', errors = 'coerce')
df['YYYYMMDDHH'] = np.where(df['YYYYMMDDHH'].dt.strftime('%H:%M') == '00:00',
                       df['YYYYMMDDHH'] + pd.DateOffset(days = 1),
                       df['YYYYMMDDHH'])
drop_columns = ['YYYYMMDD', 'HH', 'DD', 'T10N', 'TD', 'VV', 'STN']
df = df.drop(columns = drop_columns)
df.insert(0, 'YYYYMMDDHH', df.pop('YYYYMMDDHH'))
print(df.head())

# New column names 
new_column_names = { 
    'FH' : 'average windspeed (m/s)',
    'FF' : 'windspeed previous 10 minuten (m/s)',
    'FX' : 'highest windspeed (m/s)',
    'T'  : 'temperature in degrees Celcius',
    'SQ' : 'duration of sunshine (h)',
    'Q'  : 'global radiation (in J/m^2)',    
    'P'  : 'air pressure (Pa)',
    'N'  : 'cloud cover (-)',
    'U'  : 'relative humidity (in percentages)'
}
df = df.rename(columns = new_column_names)

# Group variables
windvariabelen = ['average windspeed (m/s)',
                  'windspeed previous 10 minuten (m/s)', 
                  'highest windspeed (m/s)' ]
temperature_variables = ['temperature in degrees Celcius'],
air_pressurevariabelen = ['air pressure (Pa)']
zonvariabelen = ['duration of sunshine (h)', 'global radiation (in J/m^2)']

# Physical constants
area_of_solar_panel = 2.45         ## area of one module [m^2]
efficiency_solar_panel = 0.18      ## efficiency of one solar panel [-]
number_of_solar_panels = 1          ## total number of solar panels used [-]
lat = 51.96811434560797             ## latitude of plant
lon = 4.095795682209092             ## longitude of plant
module = \
    {'theta_M': np.deg2rad(20),
     'azimuth': np.deg2rad(115)}

# Convert to S.I. units
for column in zonvariabelen:
    if column == 'duration of sunshine (h)':
        df[column] *= 0.10
    elif column == 'global radiation (in J/m^2)':
        df[column] *= 1e4
    
# Get the elevation of the module
def make_remote_request(url: str, params: dict):
    global response
    while True:
        try:
            response = requests.get(url + urllib.parse.urlencode(params))
        except (OSError, urllib3.exceptions.ProtocolError) as error:
            print(error)
            continue
        break

    return response

def elevation_function(lat, lon):
    url = 'https://api.opentopodata.org/v1/eudem25m?'
    params = {'locations': f"{lat},{lon}"}
    result = make_remote_request(url, params)
    if 'results' in result.json().keys():
        return_value = result.json()['results'][0]['elevation']
    else:
        return_value = None
    return return_value

elevation_module = elevation_function(lat, lon)
while elevation_module == None:
    time.sleep(0.40)
    elevation_module = elevation_function(lat, lon)
    
# Calculate the positions of the sun
date_time_or_doy = df['YYYYMMDDHH'] 
solpos = get_solarposition(
    time = date_time_or_doy, latitude = lat,
    longitude = lon, altitude = elevation_module,
    pressure = df['air pressure (Pa)'].values, 
    temperature = df['temperature in degrees Celcius'].values) 

# Orientation parameters
theta_M = module['theta_M']                     ## tilt angle module
az_M = module['azimuth']                        ## azimuth angle module

# Compute G_module, first decompose the ghi into dni and dhi
ghi = df['global radiation (in J/m^2)'].values  ## global horizontal irradiance in [J/m^2]
out_erbs = irradiance.erbs(ghi, solpos.zenith, solpos.index)
out_erbs = out_erbs.rename(columns = {'dni': 'dni_erbs', 'dhi': 'dhi_erbs'})
dni = out_erbs['dni_erbs'].values               ## direct normal irradiance. [J/m^2]
dhi = out_erbs['dhi_erbs'].values               ## diffuse horizontal irradiance in [J/m^2]
sky_model = 'isotropic'
poa = irradiance.get_total_irradiance(np.rad2deg(theta_M), np.rad2deg(az_M),
                         solpos.zenith, solpos.azimuth,
                         dni, ghi, dhi, dni_extra = None, airmass = None,
                         surface_type = 'sea',
                         model = sky_model)
G_module = poa['poa_global'].values

# Store into the dataframe
df['yield solar panel (kWh)'] = \
    (G_module * area_of_solar_panel * number_of_solar_panels * efficiency_solar_panel) / (1e3 * 3600 )
print("Last data frame after calculations/n")
print(df.head())
df.to_csv('~/Documents/Soleil/Lectoraar-Internship/Solar-Panel-Project/Energy/data_experimenting/df_with_calculations.csv')
# Visualisation parameters
visualisation_day = range(1, number_to_month[visualisation_month[0]][1] + 1)

df_visualisation = df[
    (df['YYYYMMDDHH'].dt.year.isin(visualisation_year)) &
    (df['YYYYMMDDHH'].dt.month.isin(visualisation_month)) &
    (df['YYYYMMDDHH'].dt.day.isin(visualisation_day))
]

# Initialize plot
fig = plt.figure(figsize = (9, 9))
ax = fig.add_subplot(1, 1, 1, projection = '3d')
plt.style.use('default')
ax.set_box_aspect((1, 1, 0.25))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.invert_yaxis()
field = 'yield solar panel (kWh)'    
str_ = field + ' in ' + number_to_month[visualisation_month[0]][0]
ax.set_title(str_)
ax.set_xlabel('hours')
ax.set_ylabel('days')
ax.set_zlabel('[kWh]')

# Plot
for day in visualisation_day:
    bools = df_visualisation['YYYYMMDDHH'].dt.day == day
    x = df_visualisation.loc[bools, 'YYYYMMDDHH'].dt.hour
    y = df_visualisation.loc[bools, 'YYYYMMDDHH'].dt.day
    z = df_visualisation.loc[bools, field]

    x_ = [list(x)[0]] + list(x) + [list(x)[-1]]
    y_ = [list(y)[0]] + list(y) + [list(y)[-1]]
    z_ = [0] + list(z) + [0]
    
    ax.plot(x_, y_, z_, 'k', linewidth = 0.1)
    
    verts = [list(zip(x_, z_))]
    poly = PolyCollection(verts, 
                              facecolors = (1, 1, 0, 0.75), 
                              edgecolors = 'k', linewidths = 1.25, closed = False)
        
    ax.add_collection3d(poly, zs = [y.values[0]], zdir = 'y') 