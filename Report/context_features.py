from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

spacing = 20
num_datapoints_context = 3

timedelta_threshold_seconds = timedelta(days=20).total_seconds()

def make_training_set(n_shifts_1, spacing, num_datapoints_context):
    filepath_train = r'../datasets/ais_train.csv'
    filepath_test = r'../datasets/ais_test.csv'

    # Load AIS historical data
    train = pd.read_csv(filepath_train, sep ='|')  # Replace with your dataset
    test = pd.read_csv(filepath_test, sep = ',')

    # Replace special values with NaN
    train['rot'] = train['rot'].replace({127: np.nan, -127: np.nan, -128: np.nan})
    train['sog'] = train['sog'].replace({102.3: np.nan})
    train['cog'] = train['cog'].replace({360: np.nan})
    train['heading'] = train['heading'].replace({511: np.nan})
    train.dropna(inplace=True)

    # Preprocessing
    train['time'] = pd.to_datetime(train['time'])
    train.sort_values(by=['vesselId', 'time'], inplace=True)
    train['isMoored'] = train['navstat']== 5
    train = train[~train['isMoored']]

    test['time'] = pd.to_datetime(test['time'])
    test.sort_values(by=['vesselId', 'time'], inplace=True)

    # Feature Engineering
    for i in range(num_datapoints_context):
        train[f'prev_lat_{i}'] = train.groupby('vesselId')['latitude'].shift(n_shifts_1 + spacing*i)
        train[f'prev_lon_{i}'] = train.groupby('vesselId')['longitude'].shift(n_shifts_1 + spacing*i)
        train[f'prev_speed_{i}'] = train.groupby('vesselId')['sog'].shift(n_shifts_1 + spacing*i)
        train[f'prev_course_{i}'] = (train.groupby('vesselId')['cog'].shift(n_shifts_1 + spacing*i) / 180) - 1        # normalized
        train[f'prev_rotation_{i}'] = train.groupby('vesselId')['rot'].shift(n_shifts_1 + spacing*i) 
        train[f'prev_heading_{i}'] = (train.groupby('vesselId')['heading'].shift(n_shifts_1 + spacing*i)/ 180) - 1 
        # Adding timedelta as a feature
        train[f'time_diff_{i}'] = train['time'].diff(n_shifts_1 + spacing*i)
        train[f'time_diff_seconds_{i}'] = train[f'time_diff_{i}'].dt.total_seconds()
        train = train[train[f'time_diff_seconds_{i}'] <= timedelta_threshold_seconds]


    train.dropna(inplace=True)


    # # Fill missing values (optional, using forward fill) Uses most recent non-null value from the row above.
    # train['prev_rotation'].fillna(method='ffill', inplace=True)

    # Drop rows with missing values
    train.dropna(inplace=True)

    print(f"Length of dataset after preprocessing: {len(train)}")
    return train
    

# train1 = make_training_set(1)
# train2 = make_training_set(2)
train3 = make_training_set(3, spacing=spacing, num_datapoints_context=num_datapoints_context)
train4 = make_training_set(4, spacing=spacing, num_datapoints_context=num_datapoints_context)
train5 = make_training_set(5, spacing=spacing, num_datapoints_context=num_datapoints_context)
train6 = make_training_set(6, spacing=spacing, num_datapoints_context=num_datapoints_context)
train7 = make_training_set(7, spacing=spacing, num_datapoints_context=num_datapoints_context)
# train8 = make_training_set(8)
# train9 = make_training_set(9)
# train10 = make_training_set(10)
# train11 = make_training_set(11)
# train12 = make_training_set(12)
# train13 = make_training_set(13)
# train14 = make_training_set(14)
# train15 = make_training_set(15)
# train16 = make_training_set(16)
# train17 = make_training_set(17)
# train18 = make_training_set(18)
# train19 = make_training_set(19)
# train20 = make_training_set(20)
train21 = make_training_set(21, spacing=spacing, num_datapoints_context=num_datapoints_context)
train22 = make_training_set(22, spacing=spacing, num_datapoints_context=num_datapoints_context)
train23 = make_training_set(23, spacing=spacing, num_datapoints_context=num_datapoints_context)
# train24 = make_training_set(24)
# train25 = make_training_set(25)
# train26 = make_training_set(26)

# train24 = make_training_set(24)

# train34 = make_training_set(34)

# train44 = make_training_set(44)

# train46 = make_training_set(46)

# train48 = make_training_set(48)

train50 = make_training_set(50, spacing=spacing, num_datapoints_context=num_datapoints_context)
train51 = make_training_set(51, spacing=spacing, num_datapoints_context=num_datapoints_context)
train52 = make_training_set(52, spacing=spacing, num_datapoints_context=num_datapoints_context)
# train53 = make_training_set(53)
# train54 = make_training_set(54)
# train55 = make_training_set(55)
# train56 = make_training_set(56)
# train57 = make_training_set(57)
# train58 = make_training_set(58)
# train59 = make_training_set(59)
# train60 = make_training_set(60)

# train62 = make_training_set(62)

# train64 = make_training_set(64)

# train74 = make_training_set(74)

# train84 = make_training_set(84)

# train60 = make_training_set(60)

# train70 = make_training_set(70)
# train71 = make_training_set(71)
# train72 = make_training_set(72)
# train73 = make_training_set(73)
# train74 = make_training_set(74)
# train75 = make_training_set(75)

# train80 = make_training_set(80)
# train81 = make_training_set(81)
# train82 = make_training_set(82)
# train83 = make_training_set(83)
# train84 = make_training_set(84)
# train85 = make_training_set(85)
# train86 = make_training_set(86)
# train87 = make_training_set(87)
# train88 = make_training_set(88)
# train89 = make_training_set(89)
# train90 = make_training_set(90)
# train91 = make_training_set(91)
# train92 = make_training_set(92)
# train93 = make_training_set(93)
# train94 = make_training_set(94)
# train95 = make_training_set(95)
# train96 = make_training_set(96)
# train97 = make_training_set(97)
# train98 = make_training_set(98)

train100 = make_training_set(100, spacing=spacing, num_datapoints_context=num_datapoints_context)
train101 = make_training_set(101, spacing=spacing, num_datapoints_context=num_datapoints_context)
train102 = make_training_set(102, spacing=spacing, num_datapoints_context=num_datapoints_context)

# train100 = make_training_set(100)
# train101 = make_training_set(101)
# train102 = make_training_set(102)
# train103 = make_training_set(103)
# train104 = make_training_set(104)
# train105 = make_training_set(105)
# train106 = make_training_set(106)
# train107 = make_training_set(107)
# train108 = make_training_set(108)
# train109 = make_training_set(109)
# train110 = make_training_set(110)
# train111 = make_training_set(111)
# train112 = make_training_set(112)
# train113 = make_training_set(113)
# train114 = make_training_set(114)
# train115 = make_training_set(115)


train150 = make_training_set(150, spacing=spacing, num_datapoints_context=num_datapoints_context)
train151 = make_training_set(151, spacing=spacing, num_datapoints_context=num_datapoints_context)
train152 = make_training_set(152, spacing=spacing, num_datapoints_context=num_datapoints_context)

# train = pd.concat([train8, train9, train10, train11, train12, train13, train14], ignore_index=True)
train = pd.concat([train3, train4, train5, train6, train7, train21, train22, train23, train50, train51, train52, train100, train101, train102, train150, train151, train152], ignore_index=True)
# train = pd.concat([train3], ignore_index=True)

feats_to_include = [
    feat 
    for i in range(num_datapoints_context) 
    for feat in (
        f'prev_lat_{i}', 
        f'prev_lon_{i}', 
        f'prev_speed_{i}', 
        f'prev_course_{i}', 
        f'prev_rotation_{i}', 
        f'prev_heading_{i}', 
        f'time_diff_seconds_{i}'
    )
]
X = train[feats_to_include]
y_lat = train['latitude']
y_lon = train['longitude']

print(f"Here comes Xlength:{len(X)}")
print(f"Here comes X.describe:\n{X.describe()}")

X_lat_train, X_lat_val, y_lat_train, y_lat_val = train_test_split(X, y_lat, test_size=0.1, random_state=42)
X_lon_train, X_lon_val, y_lon_train, y_lon_val = train_test_split(X, y_lon, test_size=0.1, random_state=42)

# Train the model
model_lat = RandomForestRegressor(n_estimators=15, verbose=3, random_state=17, warm_start=False, criterion='squared_error', max_depth=25, n_jobs=-1)
model_lat.fit(X_lat_train, y_lat_train)

model_lon = RandomForestRegressor(n_estimators=15, verbose=3, random_state=17, warm_start=False, criterion='squared_error', max_depth=25, n_jobs=-1)
model_lon.fit(X_lon_train, y_lon_train)

# Make predictions on the validation set
y_lat_pred_val = model_lat.predict(X_lat_val)
y_lon_pred_val = model_lon.predict(X_lon_val)

# Evaluate performance on the validation set
mae_lat = mean_absolute_error(y_lat_val, y_lat_pred_val)
mae_lon = mean_absolute_error(y_lon_val, y_lon_pred_val)

print(f'Mean Absolute Error for Latitude: {mae_lat}')
print(f'Mean Absolute Error for Longitude: {mae_lon}')

filepath_train = r'../datasets/ais_train.csv'

# Load AIS historical data
training_data = pd.read_csv(filepath_train, sep='|')
training_data['time'] = pd.to_datetime(training_data['time'])

# Predict future positions
def predict_future_position(id, vessel_id, time):
    # Fetch the latest known position of the vessel
    latest_data_points = training_data[training_data['vesselId'] == vessel_id]
    latest_data_points_sorted = latest_data_points.sort_values(by='time')

    # Set 'time' as the index to allow for time-based rolling window
    latest_data_points_sorted = latest_data_points_sorted.set_index('time')

    # Initialize an empty list to collect each row's data
    feature_rows = []

    # Collect feature data for each context point
    for i in range(num_datapoints_context):
        # Get the latest data point
        data_point = latest_data_points_sorted.iloc[-1 - spacing * i]

        # Prepare a dictionary with the features for the current data point
        row_data = {
            f'prev_lat_{i}': data_point['latitude'],
            f'prev_lon_{i}': data_point['longitude'],
            f'prev_speed_{i}': data_point['sog'],
            f'prev_course_{i}': (data_point['cog'] / 180) - 1,
            f'prev_rotation_{i}': data_point['rot'],
            f'prev_heading_{i}': (data_point['heading'] / 180) - 1,
            f'time_diff_seconds_{i}': (pd.to_datetime(time) - data_point.name).total_seconds()  # `data_point.name` gives the index (time)
        }

        # Add the row data to the list
        feature_rows.append(row_data)

    # Concatenate all row dictionaries into a single-row DataFrame
    featuresset = pd.DataFrame([feature_rows])

    # Flatten featuresset by converting the DataFrame to a single row with all feature columns
    featuresset_flattened = pd.concat([pd.DataFrame([row]) for row in feature_rows], axis=1)

    # Print for debugging to ensure the structure is as expected
    # print(f'Final featuresset for prediction:\n{featuresset_flattened}')

    # Make predictions
    return id, model_lat.predict(featuresset_flattened)[0], model_lon.predict(featuresset_flattened)[0]



# Open the test file for reading and the prediction file for writing
with open('../datasets/ais_test.csv', 'r') as f_test, open('../predictions/predictions_2.csv', 'w') as f_pred:
    f_pred.write("ID,longitude_predicted,latitude_predicted\n")
    for line in tqdm(f_test.readlines()[1:]):
        id, vesselID, time, scaling_factor = line.split(',')
        id, pred_lat, pred_lon = predict_future_position(id, vesselID, time)
        f_pred.write(f"{id},{pred_lon},{pred_lat}\n")