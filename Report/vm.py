import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

def make_training_set(n_shifts):
    filepath_train = '../datasets/ais_train.csv'
    filepath_test = '../datasets/ais_test.csv'

    # Load AIS historical data
    train = pd.read_csv(filepath_train, sep ='|')  # Replace with your dataset
    test = pd.read_csv(filepath_test, sep = ',')

    # Preprocessing
    train['time'] = pd.to_datetime(train['time'])
    train.sort_values(by=['vesselId', 'time'], inplace=True)
    train['isMoored'] = train['navstat']== 5
    # Why are we using the complement?
    train = train[~train['isMoored']]

    test['time'] = pd.to_datetime(test['time'])
    test.sort_values(by=['vesselId', 'time'], inplace=True)

    # Feature Engineering
    train['prev_lat'] = train.groupby('vesselId')['latitude'].shift(n_shifts)
    train['prev_lon'] = train.groupby('vesselId')['longitude'].shift(n_shifts)
    train['prev_speed'] = train.groupby('vesselId')['sog'].shift(n_shifts)
    train['prev_course'] = (train.groupby('vesselId')['cog'].shift(n_shifts) / 180) - 1        # normalized
    train['prev_rotation'] = train.groupby('vesselId')['rot'].shift(n_shifts) 
    train['prev_heading'] = (train.groupby('vesselId')['heading'].shift(n_shifts)/ 180) - 1 
    train['hour'] = train['time'].dt.hour
    # Could change this to is_weekend
    train['day_of_week'] = train['time'].dt.dayofweek
    # Adding timedelta as a feature
    train['time_diff'] = train['time'].diff(n_shifts)
    train['time_diff_seconds'] = train['time_diff'].dt.total_seconds()

    # --------------------------------- prev_rot-related stuff
    # Replace special values with NaN
    train['prev_rotation'] = train['prev_rotation'].replace({127: np.nan, -127: np.nan, -128: np.nan})
    # Optional: Create a new column for turn information availability
    train['turn_info_available'] = np.where(train['prev_rotation'] == -128, 0, 1)

    # Create binary columns for turn direction and magnitude
    train['turn_direction'] = np.where(train['prev_rotation'] > 0, 'right', 'left')
    train['turn_magnitude'] = train['prev_rotation'].abs()

    # Fill missing values (optional, using forward fill) Uses most recent non-null value from the row above.
    train['prev_rotation'].fillna(method='ffill', inplace=True)

    # Drop rows with missing values
    train.dropna(inplace=True)

    print(f"Length of dataset after preprocessing: {len(train)}")
    return train
    

train1 = make_training_set(1)
train2 = make_training_set(2)
train3 = make_training_set(3)
train4 = make_training_set(4)
train5 = make_training_set(5)
train6 = make_training_set(6)
train7 = make_training_set(7)

train = pd.concat([train1, train2, train3, train4, train5, train6, train7], ignore_index=True)

feats_to_include = ['prev_lat', 'prev_lon', 'prev_speed', 'prev_course','prev_rotation', 'turn_info_available', 'prev_heading', 'time_diff_seconds']
X = train[feats_to_include]
y_lat = train['latitude']
y_lon = train['longitude']

X_lat_train, X_lat_val, y_lat_train, y_lat_val = train_test_split(X, y_lat, test_size=0.1, random_state=42)
X_lon_train, X_lon_val, y_lon_train, y_lon_val = train_test_split(X, y_lon, test_size=0.1, random_state=42)

# Train the model
model_lat = RandomForestRegressor(n_estimators=10, verbose=3, random_state=42)
model_lat.fit(X_lat_train.values, y_lat_train.values)

model_lon = RandomForestRegressor(n_estimators=10, verbose=3, random_state=42)
model_lon.fit(X_lon_train.values, y_lon_train.values)

# Make predictions on the validation set
y_lat_pred_val = model_lat.predict(X_lat_val)
y_lon_pred_val = model_lon.predict(X_lon_val)

# Evaluate performance on the validation set
mae_lat = mean_absolute_error(y_lat_val, y_lat_pred_val)
mae_lon = mean_absolute_error(y_lon_val, y_lon_pred_val)

print(f'Mean Absolute Error for Latitude: {mae_lat}')
print(f'Mean Absolute Error for Longitude: {mae_lon}')

from datetime import datetime

filepath_train = '../datasets/ais_train.csv'

# Load AIS historical data
training_data = pd.read_csv(filepath_train, sep ='|')  # Replace with your dataset

# Predict future positions
def predict_future_position(id, vessel_id, time):
    # Fetch the latest known position of the vessel
    latest_data = training_data[training_data['vesselId'] == vessel_id].sort_values(by='time').iloc[-1]

    new_data = {
        'prev_lat': latest_data['latitude'],
        'prev_lon': latest_data['longitude'],
        'prev_speed': latest_data['sog'],
        'prev_course': (latest_data['cog'] / 180) - 1,
        'prev_rotation': latest_data['rot'],
        'turn_info_available': 1 if latest_data['rot'] != -128 else 0,
        'prev_heading' : (latest_data['heading'] / 180) - 1,

        # Convert the times to a datetime_obj
        'time_diff_seconds' : (datetime.strptime(time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(latest_data['time'], '%Y-%m-%d %H:%M:%S')).total_seconds(),
    }
    # why is it done with all this list stuff?
    return id, model_lat.predict([list(new_data.values())])[0], model_lon.predict([list(new_data.values())])[0]
    # return id, model_lat.predict([list(new_data.values())]), model_lon.predict([list(new_data.values())]), new_data['time_diff_seconds'], latest_data['time']

# ['prev_lat', 'prev_lon', 'prev_speed', 'prev_course','prev_rotation', 'turn_info_available', 'prev_heading', 'time_diff_seconds']

from tqdm import tqdm
# Open the test file for reading and the prediction file for writing
with open('../datasets/ais_test.csv', 'r') as f_test, open('../predictions/predictions_2.csv', 'w') as f_pred:
    f_pred.write("ID,longitude_predicted,latitude_predicted\n")
    for line in f_test.readlines()[1:]:
        id, vesselID, time, scaling_factor = line.split(',')
        id, pred_lat, pred_lon = predict_future_position(id, vesselID, time)
        f_pred.write(f"{id},{pred_lon},{pred_lat}\n")