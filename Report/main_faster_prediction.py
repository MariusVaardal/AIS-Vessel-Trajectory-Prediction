from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

random_seed = 17

timedelta_threshold_seconds = timedelta(days=20).total_seconds()

def make_training_set(n_shifts):
    filepath_train = r'../datasets/ais_train.csv'
    filepath_test = r'../datasets/ais_test.csv'

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
    # Adding timedelta as a feature
    train['time_diff'] = train['time'].diff(n_shifts)
    train['time_diff_seconds'] = train['time_diff'].dt.total_seconds()
    # Apply the moving average function to each vessel group
    train.dropna(inplace=True)

    # --------------------------------- prev_rot-related stuff
    # Replace special values with NaN
    train['prev_rotation'] = train['prev_rotation'].replace({127: np.nan, -127: np.nan, -128: np.nan})
    train['prev_speed'] = train['prev_speed'].replace({102.3: np.nan})
    train['prev_course'] = train['prev_course'].replace({360: np.nan})
    train['prev_heading'] = train['prev_heading'].replace({511: np.nan})
    train.dropna(inplace=True)

    train = train[train['time_diff_seconds'] <= timedelta_threshold_seconds]

    # # Fill missing values (optional, using forward fill) Uses most recent non-null value from the row above.
    # train['prev_rotation'].fillna(method='ffill', inplace=True)

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
train8 = make_training_set(8)
train9 = make_training_set(9)
train10 = make_training_set(10)
train11 = make_training_set(11)
train12 = make_training_set(12)
train13 = make_training_set(13)
train14 = make_training_set(14)
# train15 = make_training_set(15)
# train16 = make_training_set(16)
# train17 = make_training_set(17)
# train18 = make_training_set(18)
# train19 = make_training_set(19)
# train20 = make_training_set(20)
# train21 = make_training_set(21)
# train22 = make_training_set(22)
# train23 = make_training_set(23)
# train24 = make_training_set(24)
# train25 = make_training_set(25)
# train26 = make_training_set(26)

# train24 = make_training_set(24)

train34 = make_training_set(34)

train44 = make_training_set(44)

train46 = make_training_set(46)

train48 = make_training_set(48)

train50 = make_training_set(50)
# train51 = make_training_set(51)
train52 = make_training_set(52)
# train53 = make_training_set(53)
train54 = make_training_set(54)
train55 = make_training_set(55)
train56 = make_training_set(56)
# train57 = make_training_set(57)
train58 = make_training_set(58)
# train59 = make_training_set(59)
train60 = make_training_set(60)

train62 = make_training_set(62)

train64 = make_training_set(64)

train74 = make_training_set(74)

train84 = make_training_set(84)

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

# train100 = make_training_set(100)
# train101 = make_training_set(101)
# train102 = make_training_set(102)

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

# train = pd.concat([train8, train9, train10, train11, train12, train13, train14], ignore_index=True)
train = pd.concat([train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12, train13, train14, train34, train44, train46, train48, train50, train52, train54, train55, train56, train58, train60, train62, train64, train74, train84], ignore_index=True)

feats_to_include = ['prev_lat', 'prev_lon', 'prev_speed', 'prev_course','prev_rotation', 'prev_heading', 'time_diff_seconds']
X = train[feats_to_include]
print(f"Length of X: {len(X)}")
y = train[['longitude', 'latitude']]
print(f"shape of y: {np.shape(y)}")

print(f"Here comes X.describe:\n{X.describe()}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=random_seed)

# Train the model
model = RandomForestRegressor(n_estimators=1, verbose=3, random_state=random_seed, warm_start=False, criterion='squared_error', max_depth=25, n_jobs=-1)
model.fit(X_train.values, y_train.values)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

print(f"Length of X_val: {len(X_val)}")
print(f"y pred validation shape: {np.shape(y_pred_val)}")

# Evaluate performance on the validation set
mae = mean_absolute_error(y_val, y_pred_val)

print(f'Mean Absolute Error for lon and lat: {mae}')

filepath_train = r'../datasets/ais_train.csv'

# Load AIS historical data
training_data = pd.read_csv(filepath_train, sep='|')
training_data['time'] = pd.to_datetime(training_data['time'])


def make_prediction_set_line(vessel_id, time):
    # Fetch the latest known position of the vessel
    latest_data_points = training_data[training_data['vesselId'] == vessel_id]
    latest_data_points_sorted = latest_data_points.sort_values(by='time')
    
    # Set 'time' as the index to allow for time-based rolling window
    latest_data_points_sorted = latest_data_points_sorted.set_index('time')   
    # Get the latest data point
    latest_data_point = latest_data_points_sorted.iloc[-1]

    # Prepare the new data for prediction
    new_data = {
        'prev_lat': latest_data_point['latitude'],
        'prev_lon': latest_data_point['longitude'],
        'prev_speed': latest_data_point['sog'],
        'prev_course': (latest_data_point['cog'] / 180) - 1,
        'prev_rotation': latest_data_point['rot'],
        'prev_heading': (latest_data_point['heading'] / 180) - 1,

        # Use the datetime objects for the time difference
        'time_diff_seconds': (pd.to_datetime(time) - latest_data_point.name).total_seconds(),  # .name gives the index (time)
    }

    # Make predictions
    return list(new_data.values())

# Open the test file for reading and the prediction file for writing
prediction_set = []
ids = []
with open('../datasets/ais_test.csv', 'r') as f_test:
    for line in tqdm(f_test.readlines()[1:]):
        id, vesselID, time, scaling_factor = line.split(',')
        ids.append(id)
        prediction_set_line = make_prediction_set_line(vesselID, time)
        prediction_set.append(prediction_set_line)

prediction_set_np = np.array(prediction_set)
print(f"Shape of prediction set: {prediction_set.shape}")

predictions = model.predict(prediction_set)
print(f"Shape of predictions: {np.shape(predictions)}")



