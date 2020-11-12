from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras import regularizers
from keras.callbacks import ModelCheckpoint

x_scaler = MinMaxScaler(feature_range=(0, 10))
y_scaler = MinMaxScaler(feature_range=(0, 10))
tr_scaler = MinMaxScaler(feature_range=(0, 10))
trend = ["trend", "daily", "weekly", "yearly"]

def build_lstm_model(l_num=1, f_num=5, timesteps=32, batch_size=64, units=100, dropout=False, optimizer='adam', loss='mean_squared_error',  regularizer=None):
                     
    model = Sequential(name="LSTM_model-%d-%d-%d-%d-%d" % (l_num, f_num, timesteps, batch_size, units))
    model.add(Input(shape=(timesteps, f_num)))
    for i in range(l_num):               
        model.add(LSTM(units, return_sequences=True if i!=l_num-1 else False, kernel_regularizer=regularizer))         
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    
    return model


def create_training_data(data, timesteps, features_price=["Open", "Close", "High", "Low"], features_trend = ["Volume"], column="Close"):
    """
    Creates X and Y data
    :param data: Input dataset with forex data
    :param timesteps: Lookback for dataset in list
    :returns X, y: Numpy arrays, containing training data and answers
    """

    X_data = x_scaler.fit_transform(data[features_price].to_numpy().reshape(-1,1))
    X_data = X_data.reshape(-1, len(features_price))
    X_data = np.append(X_data, x_scaler.fit_transform(data[features_trend].to_numpy()), axis=1)
    y_data = y_scaler.fit_transform(data[column].to_numpy().reshape(-1,1))

    X, y = [], []
    for i in range(len(X_data)-timesteps):
        X.append(X_data[i:(i+timesteps), :])
        y.append(y_data[i + timesteps])   
    np.reshape(X, (-1, timesteps, len(features_price) + len(features_trend)))
    
    return np.array(X), np.array(y)


def denormalize_training_data(y):
    return y_scaler.inverse_transform(y)


def validate_model(dataframe, X_test, y_test, model, column):
    y_pred = model.predict(X_test)
    y_pred = denormalize_training_data(y_pred)
    y_test = denormalize_training_data(y_test)
    print("Absolute error mean is %.6f" % (mean_absolute_error(y_pred, y_test)))
    graph_error_mean_per_hour(dataframe, y_pred, column)


def test_model(data, model, timesteps, column="Close", features_price=["Open", "Close", "High", "Low"], \
               features_trend = ["Volume"], split=True):
    X, y = create_training_data(data, timesteps, features_price, features_trend, column)
    if split:
        X_train, X, y_train, y = train_test_split(X, y, test_size=0.1, shuffle=False)
    y_pred = model.predict(X)
    y_pred = denormalize_training_data(y_pred)
    y_test = denormalize_training_data(y)
    print("Absolute error mean is %.6f" % (mean_absolute_error(y_pred, y_test)))
    graph_error_mean_per_hour(data, y_pred, column)
    
    
def train_model(model, data, timesteps, batch_size, epochs, features, features_rest=None, 
                shuffle=True, validation_data=None, save_best_only=False, column="Close", pr=False, val=False):
    
    st = datetime.now()
    if pr:
        X, y = create_training_data_with_pr(data, timesteps, features, column)
    else:
        X, y = create_training_data(data, timesteps, features, column, features_rest)   


    
    if validation_data is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    else:
        X_train, y_train = X, y
        if pr:
            X_test, y_test = create_training_data_with_pr(validation_data, timesteps, features, column)   
        else:
            X_test, y_test = create_training_data(validation_data, timesteps, features, column, features_rest)
        
    file_path = model_path + model.name + "_weights.hdf5"
    checkpointer = ModelCheckpoint(filepath = file_path, save_best_only=save_best_only)
                     
    for i in range(1, epochs+1):
        print("Epoch №%d\n" % i)
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test),\
                  shuffle=shuffle, callbacks=[checkpointer], verbose=1)
        if val:
            validate_model(validation_data if validation_data is not None else data, X_test, y_test, model, column) 
        

    delta = int((datetime.now()-st).total_seconds())
    print("Training took %d minutes and %d seconds\n" % (int(delta/60), int(delta%60)))
    
    return model