import tkinter as tk 
import sys
import ctypes
import itertools
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import csv
import arcpy
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
import tkinter as tk 
import itertools
import math
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.dates as mdates
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import matplotlib
import copy
from matplotlib import *
import pylab as pl
from tkinter import Button, Toplevel
from tensorflow.keras.layers import LSTMCell, RNN, Dense
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import (Input, Dense, MultiHeadAttention, LayerNormalization, Add, Dropout,LSTM, LSTMCell, RNN, Concatenate, Flatten, Reshape)
#################################################################### ADD MISSED PACKAGES
# Process: Define the parameters of the toolbox
file = arcpy.GetParameterAsText(0)
isCNR= arcpy.GetParameterAsText(1)
isIRREG= arcpy.GetParameterAsText(2)
path = arcpy.GetParameterAsText(3)
latmin = float(arcpy.GetParameterAsText(4))
latmax = float(arcpy.GetParameterAsText(5))
lonmin = float(arcpy.GetParameterAsText(6))
lonmax = float(arcpy.GetParameterAsText(7))
steps = int(arcpy.GetParameterAsText(8))
nod = arcpy.GetParameterAsText(9)
nod1 = arcpy.GetParameterAsText(10)#################
LR= arcpy.GetParameterAsText(11)
epo= arcpy.GetParameterAsText(12)
ischecked = arcpy.GetParameterAsText(13)
####################################################################
# Process: Control some of the LSTM model hyperparameters 
if str(isIRREG) == 'false':
    if nod == '':
        nodi = 50 if steps == 1 else 100
    else:
        nodi = int(nod)

    if steps > 1:
        if nod1 == '':
            nodi1 = 50
        else:
            nodi1 = int(nod1)

    if LR == '':
        learnR = 0.001 if steps == 1 else 0.0001
    else:
        learnR = float(LR)

    if epo == '':
        epochs = 15 if steps == 1 else 35
    else:
        epochs = int(epo)

else:
    if nod == '':
        nodi = 50 if steps == 1 else 8
    else:
        nodi = int(nod)

    if LR == '':
        learnR = 0.001
    else:
        learnR = float(LR)

    if epo == '':
        epochs = 35 if steps == 1 else 50
    else:
        epochs = int(epo)
#################################################################### 
if str(isCNR) == 'true':
    with open(file, 'r') as my_file:
        reader = csv.reader(my_file)
        rows = list(reader)

        # Process: Check if the timesteps are regular or not 
        a = [l for l in rows if any(s.startswith('List') for s in l)]
        a1 = [item for sublist in a for item in sublist]
        b = rows.index(a1)
        list_dates = rows[b]
        list_dates = list(map(lambda x: x[:-10], list_dates))
        lista = list_dates[0]
        list_dates.remove(list_dates[0])
        list_dates = list(map(lambda x: x[1:], list_dates))
        list_dates.insert(0, lista[15:])
        
        tm1 = list_dates
        tm2 = list_dates[:-1]
        tm2.insert(0, tm2[0])
        tms = []
        
        for i in range(len(tm1)):
            d1 = datetime.strptime(tm1[i], "%Y-%m-%d")
            d2 = datetime.strptime(tm2[i], "%Y-%m-%d")
            td = d2 - d1
            td_in_sec = int(td.total_seconds() / 86400)
            tms.append(td_in_sec)
        
        tms_ar = np.array(tms)
        tms_u = np.unique(tms_ar)
        disp = rows[43:]

else:
    with open(file, "r") as file1:
        dati = list(csv.reader(file1, delimiter=","))
    
    disp = dati[43:]
    a = [l for l in dati if any(s.startswith('ID') for s in l)]
    a1 = [item for sublist in a for item in sublist]
    b = dati.index(a1)
    list_dates = dati[b]
    list_dates = list_dates[3:]
    
    tm1 = list_dates
    tm2 = list_dates[:-1]
    tm2.insert(0, tm2[0])
    tms = []
    
    for i in range(len(tm1)):
        d1 = datetime.strptime(tm1[i], "%d-%b-%Y")
        d2 = datetime.strptime(tm2[i], "%d-%b-%Y")
        td = d2 - d1
        td_in_sec = int(td.total_seconds() / 86400)
        tms.append(td_in_sec)
    
    tms_ar = np.array(tms)
    tms_u = np.unique(tms_ar)
####################################################################
##if len(tms_u)!=2:
##   arcpy.AddError("Time series forcasting using LSTM model requires no data gaps.\n You need to fill missing values as part of the data analysis and cleaning process before using this toolbox.") 
##   sys.exit(0)
####################################################################
# Process: Define the coordinates of the selected area to be trained 
displacements=[]

for i in range(len(disp)):
    if np.float64(disp[i][1]) < latmax and np.float64(disp[i][1]) > latmin:
        if np.float64(disp[i][2]) < lonmax and np.float64(disp[i][2]) > lonmin:
            displacements.append(disp[i])

Lat = list(map(lambda x: x[1], displacements))
Lon = list(map(lambda x: x[2], displacements))

Lat_df = pd.DataFrame(Lat, columns=['y'])
Lon_df = pd.DataFrame(Lon, columns=['x'])

df = pd.concat([Lon_df, Lat_df], axis=1)
####################################################################
if str(isCNR) == 'true':
    displacements = list(map(lambda x: x[9:], displacements))
else:
    displacements = list(map(lambda x: x[3:], displacements))
####################################################################
def add_days_to_last_date(dates, period):
    """Add a fixed number of days to the last date in a list."""
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    new_date = last_date + timedelta(days=period)
    dates.append(new_date.strftime("%Y-%m-%d"))
    return dates

def add_days_to_last_date_Mult(dates, periods):
    """Add multiple periods to the last date in sequence."""
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    for period in periods:
        period_int = int(period)  # Convert numpy.int64 to native Python int
        new_date = last_date + timedelta(days=period_int)
        dates.append(new_date.strftime("%Y-%m-%d"))
        last_date = new_date
    return dates

# === Data Preprocessing for One-Step Prediction ===

def df_to_X_y(df, window_size):
    """Reshape dataset for 1-step prediction."""
    df_as_np = df.to_numpy()
    X, y = [], []
    for i in range(df_as_np.shape[1]):
        row = [[a] for a in df_as_np[:-1, i]]
        X.append(row)
        label = df_as_np[window_size, i]
        y.append(label)
    return np.array(X), np.array(y)

# === Accuracy Plotting ===

def plotCurves(acc, val_acc, epochs, RMSE):
    """Plot training and validation accuracy."""
    numbers = list(range(1, epochs + 1)) 
    plt.plot(numbers, acc, 'y', label='Training acc')
    plt.plot(numbers, val_acc, 'r', label='Validation acc')
    plt.title(f'Training and Validation Accuracy\nRMSE = {RMSE}')
    plt.xlabel('Epochs')
    plt.ylabel('Root Mean Squared Error')
    plt.legend()
    plt.show()

# === Data Preprocessing for Multi-Step Prediction ===

def df_to_XY_N(df, WIN_SIZE, st):
    """Reshape dataset for N-step prediction."""
    df_as_np = df.to_numpy()
    X, y = [], []
    for i in range(df_as_np.shape[1]):
        row = [[a] for a in df_as_np[:-st, i]]
        X.append(row)
        label = df_as_np[WIN_SIZE:WIN_SIZE + st, i]
        y.append(label)
    return np.array(X), np.array(y)

# === TG-LSTM Model ===

class TimeGatedLSTMCell(LSTMCell):
    def __init__(self, units, **kwargs):
        super(TimeGatedLSTMCell, self).__init__(units, **kwargs)

    def build(self, input_shape):
        features = input_shape[-1] - 1
        super().build((features,))
        self.time_kernel = self.add_weight(shape=(1, self.units * 4), name='time_kernel', initializer='uniform')
        self.time_bias = self.add_weight(shape=(self.units * 4,), name='time_bias', initializer='zeros')
        self.built = True

    def call(self, inputs, states, training=None):
        features, time_info = inputs[:, :-1], inputs[:, -1:]
        lstm_output, new_states = super().call(features, states, training)
        time_gating_vector = tf.matmul(tf.expand_dims(time_info, -1), self.time_kernel) + self.time_bias
        time_gating_vector = tf.reshape(time_gating_vector, [-1, 4 * self.units])
        i, f, c, o = tf.split(time_gating_vector, 4, axis=1)
        c = f * new_states[1] + i * self.activation(c)
        h = o * self.activation(c)
        return h, [h, c]

def build_time_gated_lstm_model(input_shape, units, learn_rate):
    model = Sequential([
        RNN(TimeGatedLSTMCell(units), input_shape=input_shape),
        Dense(2, activation='linear')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learn_rate), metrics=[RootMeanSquaredError()])
    return model

# === Temporal Fusion Transformer Model ===

def temporal_fusion_transformer(input_shape, output_steps, d_model=8, num_heads=2):
    """
    Build a simplified TFT model using Keras layers.
    
    Parameters:
    - input_shape: Shape of the input data (time_steps, num_features)
    - output_steps: Number of time steps to predict
    - d_model: Dimensionality of model layers
    - num_heads: Number of attention heads
    
    Returns:
    - Keras Model instance
    """
    input_layer = Input(shape=input_shape)
    lstm_out = LSTM(units=d_model, return_sequences=True)(input_layer)
    
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(lstm_out, lstm_out)
    attn_output = LayerNormalization(epsilon=1e-5)(attn_output + lstm_out)

    grn_output = Dense(d_model, activation='relu')(attn_output)
    grn_output = LayerNormalization(epsilon=1e-5)(grn_output + attn_output)

    flattened_output = Flatten()(grn_output)
    output_layer = Dense(output_steps * 2)(flattened_output)
    output_layer = Reshape((output_steps, 2))(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# === Custom Hybrid Loss Function ===

def custom_l1_l2_loss(alpha=0.3, beta=0.7):
    """Create a custom loss function combining L1 and L2 losses."""
    @tf.autograph.experimental.do_not_convert
    def loss(y_true, y_pred):
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return alpha * l1_loss + beta * l2_loss
    return loss

#################################################################### 
# Process: Define the functions of the tkinter 


def show():
    p = arcpy.mp.ArcGISProject('current')
    m = p.listMaps()[0]
    lyrFile = m.listLayers()[0]
    desc = arcpy.Describe(lyrFile)
    my_string = desc.FIDSet
    my_list = my_string.split(";") 

    if len(my_list) != 1:
        MessageBox = ctypes.windll.user32.MessageBoxW
        MessageBox(None, 'You are allowed to predict only one point. More than one point has been selected!', 'Window title', 0)
    else:
        e = int(my_string)
        date_range = [pd.to_datetime(date) for date in list_dates]
        te_X = np.append(Xe1[e], Xe[e])

        if len(tms_u) == 2:
            interval = pd.to_datetime(list_dates[1]) - pd.to_datetime(list_dates[0])
            last_date = pd.to_datetime(list_dates[-1])
            new_dates = [(last_date + i * interval).strftime('%Y-%m-%d') for i in range(1, steps + 1)]        
            list_datesNew = copy.deepcopy(list_dates)
            list_datesNew.extend(new_dates)        
            date_range_predi = [pd.to_datetime(date1) for date1 in list_datesNew]
            te_Xe = np.append(te_X, test_predictFIN[e])
        else:
            if steps == 1:
                period = abs(int(test_predictFIN[e, 1]))
                list_datesNew = copy.deepcopy(list_dates)
                date_range_pred = add_days_to_last_date(list_datesNew, period)
                date_range_predi = [pd.to_datetime(date) for date in date_range_pred]
                te_Xe = np.append(te_X, test_predictFIN[e, 0])
            else:
                periods = abs(test_predictFIN[e, :, 1].astype(int))
                list_datesNew = copy.deepcopy(list_dates)
                date_range_pred = add_days_to_last_date_Mult(list_datesNew, periods)
                date_range_predi = [pd.to_datetime(date) for date in date_range_pred]
                te_Xe = np.append(te_X, test_predictFIN[e, :, 0])

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle('Predicting', fontsize=14)
        ax.set_xlabel('Time series Date', fontsize=14)
        ax.set_ylabel('Time series Displacements of The Point {} \n '.format(e), fontsize=14)

        ax.plot(date_range_predi, te_Xe, label='Predicted', linestyle='dashed', color='blue')
        ax.plot(date_range, te_X, label='Original', color='red')

        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.close()
def Close():
    root.destroy()
#########################################################################
# Process: Make directoty folder
directory = "output"
outputFolder = os.path.join(path, directory)
os.mkdir(outputFolder)
# Process: Make XY Event Layer and add it to ArcMap
df.to_csv(path+'\XYdata.csv')
arcpy.env.workspace = path
arcpy.env.overwriteOutput = True
Input_Table = path+'\XYdata.csv'
Output_Workspace = path
out_Layer = "XYpoints_layer"
XY_Event_Layer = Input_Table
X_Cord = "x"
Y_Cord = "y"
spRef = arcpy.SpatialReference(4326)
saved_Layer = path+"\output\XYpoints.lyrx"
arcpy.MakeXYEventLayer_management(Input_Table, X_Cord, Y_Cord, out_Layer, spRef)
arcpy.management.SaveToLayerFile(out_Layer, saved_Layer)
inFeatures = saved_Layer
outLocation = path+'\output'
outFeatureClass = 'XYpointsShP'
lyrFile=arcpy.FeatureClassToFeatureClass_conversion(inFeatures,outLocation,outFeatureClass)
shp_path = outLocation+ '\\' + outFeatureClass + ".shp"
aprx = arcpy.mp.ArcGISProject("CURRENT")
map = aprx.listMaps()[0]
map.addDataFromPath(shp_path)
num_cols = len(list_dates)
rng = range(1, num_cols+1)
new_cols = ['X' + str(i) for i in rng]
# ensure the length of the new columns list is equal to the length of df's columns
disp_df = pd.DataFrame(displacements,columns=new_cols)
num_cols1 = len(displacements)
rng1 = range(1, num_cols1+1)
new_cols1 = ['X' + str(i) for i in rng1]
val=disp_df.values.T
disp_df1 = pd.DataFrame(val,columns=new_cols1)
####################################################################
# === Validate Displacement Count ===
if len(displacements) > 100000:
    arcpy.AddError("The number of displacement samples is not allowed to exceed 100000.\nChange the extent of the area or recheck your dataset.")
    sys.exit(0)

# === TRAINING PART ===
if str(isIRREG) == 'false':
    
    # === ONE-STEP PREDICTION ===
    if steps == 1:
        WINDOW_SIZE = win1st
        X1, y1 = df_to_X_y(disp_df1, WINDOW_SIZE)
        X1 = X1.astype('float64')
        y1 = y1.astype('float64')

        vt = int(0.6 * len(displacements))
        ve = int(0.2 * len(displacements))

        X_train1, y_train1 = X1[:vt], y1[:vt]
        X_val1, y_val1 = X1[vt:vt+ve], y1[vt:vt+ve]
        X_test1, y_test1 = X1[vt+ve:vt+2*ve], y1[vt+ve:vt+2*ve]

        model1 = Sequential()
        model1.add(InputLayer((win1st, 1)))  # One input feature
        model1.add(LSTM(nodi))
        model1.add(Dense(1, 'linear'))
        model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learnR), metrics=[RootMeanSquaredError()])
        model1.summary()

        history = model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=epochs)

        acc = history.history['root_mean_squared_error']
        val_acc = history.history['val_root_mean_squared_error']
        test_predictions = model1.predict(X_test1).flatten()
        RMSE = round(math.sqrt(mean_squared_error(y_test1, test_predictions)), 3)

        MessageBox = ctypes.windll.user32.MessageBoxW
        MessageBox(None, f'The dataset has been trained and tested.\nThe Root Mean Squared Error is {RMSE} \n', 'Window title', 0)

        if str(ischecked) == 'true':
            plotCurves(acc, val_acc, epochs, RMSE)

        # === Add TKINTER Prediction Box ===
        dd_as_np = disp_df1.to_numpy()
        Xe, Xe1 = [], []

        for j in range(dd_as_np.shape[1]):
            row = [[a] for a in dd_as_np[1:, j]]
            Xe.append(row)
            Xe1.append(dd_as_np[0, j])

        Xe = np.array(Xe).astype('float64')
        Xe1 = np.array(Xe1).astype('float64')
        test_predictFIN = model1.predict(Xe)

        root = tk.Tk()
        root.geometry("200x150")
        tk.Button(root, text='Another Point', command=show).grid(row=1, column=0, padx=50, pady=20)
        tk.Button(root, text="Exit", command=Close).grid(row=2, column=0, padx=50, pady=10)
        root.mainloop()

    # === MULTI-STEP PREDICTION ===
    elif steps > 1:
        st = steps
        WIN_SIZE = len(list_dates) - st
        X1, y1 = df_to_XY_N(disp_df1, WIN_SIZE, st)
        X1 = X1.astype('float64')
        y1 = y1.astype('float64')

        vt = int(0.6 * len(displacements))
        ve = int(0.2 * len(displacements))

        X_train2, y_train2 = X1[:vt], y1[:vt]
        X_val2, y_val2 = X1[vt:vt+ve], y1[vt:vt+ve]
        X_test2, y_test2 = X1[vt+ve:vt+2*ve], y1[vt+ve:vt+2*ve]

        model2 = Sequential()
        model2.add(LSTM(nodi, input_shape=(WIN_SIZE, 1), return_sequences=True))
        model2.add(LSTM(nodi1, return_sequences=False))
        model2.add(Dense(st, 'linear'))
        model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learnR), metrics=[RootMeanSquaredError()])
        model2.summary()

        cp2 = ModelCheckpoint('model2/', save_best_only=True)
        history2 = model2.fit(X_train2, y_train2, validation_data=(X_val2, y_val2), epochs=epochs, callbacks=[cp2])

        acc = history2.history['root_mean_squared_error']
        val_acc = history2.history['val_root_mean_squared_error']
        test_predictions2 = model2.predict(X_test2)
        RMSE = round(math.sqrt(mean_squared_error(y_test2, test_predictions2)), 3)

        MessageBox = ctypes.windll.user32.MessageBoxW
        MessageBox(None, f'The dataset has been trained and tested.\nThe Root Mean Squared Error is {RMSE} \n', 'Window title', 0)

        if str(ischecked) == 'true':
            plotCurves(acc, val_acc, epochs, RMSE)

        # === Add TKINTER Prediction Box ===
        dd_as_np = disp_df1.to_numpy()
        Xe, Xe1 = [], []

        for j in range(dd_as_np.shape[1]):
            row = [[a] for a in dd_as_np[st:, j]]
            Xe.append(row)
            Xe1.append(dd_as_np[:st, j])

        Xe = np.array(Xe).astype('float64')
        Xe1 = np.array(Xe1).astype('float64')
        test_predictFIN = model2.predict(Xe)

        root = tk.Tk()
        root.geometry("200x150")
        tk.Button(root, text='Another Point', command=show).grid(row=1, column=0, padx=50, pady=20)
        tk.Button(root, text="Exit", command=Close).grid(row=2, column=0, padx=50, pady=10)
        root.mainloop()

    # === INVALID STEPS HANDLING ===
    else:
        arcpy.AddError("The number of predicted time steps is not valid.\nRefill the field of this parameter with a valid value.")
        sys.exit(0)

if str(isIRREG) == 'true':

    # === MULTI-STEP PREDICTION ===
    if steps > 1:
        epochs = 50
        time_steps = len(tms_ar) - steps
        num_features = 2
        output_steps = steps
        st = steps
        WIN_SIZE = len(list_dates) - st

        # Prepare dataset
        X1, y1 = df_to_XY_N(disp_df1, WIN_SIZE, st)
        X1 = X1.astype('float64')
        y1 = y1.astype('float64')
        X1 = np.reshape(X1, (X1.shape[0], X1.shape[1]))

        # Time intervals
        tmdomX = np.reshape(tms_ar[:-st], (1, -1, 1))
        tmdomX = np.repeat(tmdomX, repeats=X1.shape[0], axis=0)
        tmdomX = np.reshape(tmdomX, (tmdomX.shape[0], tmdomX.shape[1]))

        tmdomY = np.reshape(tms_ar[-st:], (1, -1, 1))
        tmdomY = np.repeat(tmdomY, repeats=y1.shape[0], axis=0)
        tmdomY = np.reshape(tmdomY, (tmdomY.shape[0], tmdomY.shape[1]))

        # Stack with displacement data
        X1 = np.stack((X1, tmdomX), axis=-1)
        y1 = np.stack((y1, tmdomY), axis=-1)

        # Train/validation/test split
        vt = int(0.6 * len(displacements))
        ve = int(0.2 * len(displacements))
        X_train, y_train = X1[:vt], y1[:vt]
        X_val, y_val = X1[vt:vt+ve], y1[vt:vt+ve]
        X_test, y_test = X1[vt+ve:vt+2*ve], y1[vt+ve:vt+2*ve]

        tft_model = temporal_fusion_transformer(input_shape=(time_steps, num_features), output_steps=output_steps)
        optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        tft_model.compile(optimizer=optimizer, loss=custom_l1_l2_loss(alpha=0.3, beta=0.7), metrics=[RootMeanSquaredError()])
        history = tft_model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_data=(X_val, y_val))

        acc = history.history['root_mean_squared_error']
        val_acc = history.history['val_root_mean_squared_error']
        test_predictions2 = tft_model.predict(X_test)

        MSE = mean_squared_error(y_test[:, :, 0], test_predictions2[:, :, 0])
        RMSE = round(math.sqrt(MSE), 3)

        ctypes.windll.user32.MessageBoxW(None, f'The dataset has been trained and tested.\nThe Root Mean Squared Error is {RMSE}', 'Window title', 0)
        if str(ischecked) == 'true':
            plotCurves(acc, val_acc, epochs, RMSE)

        # Interactive TK GUI
        dd_as_np = disp_df1.to_numpy()
        Xe, Xe1 = [], []
        for j in range(dd_as_np.shape[1]):
            Xe.append([[a] for a in dd_as_np[st:, j]])
            Xe1.append(dd_as_np[:st, j])
        Xe = np.array(Xe).astype('float64')
        Xe1 = np.array(Xe1).astype('float64')

        tmdomXFIN = tms_ar[st:]
        tmdomXFIN = np.reshape(tmdomXFIN, (1, -1, 1))
        tmdomXFIN = np.repeat(tmdomXFIN, repeats=Xe.shape[0], axis=0)
        tmdomXFIN = np.reshape(tmdomXFIN, (tmdomXFIN.shape[0], tmdomXFIN.shape[1]))

        XeFIN = np.stack((Xe[:, :, 0], tmdomXFIN), axis=-1)
        test_predictFIN = tft_model.predict(XeFIN)

        root = tk.Tk()
        root.geometry("200x150")
        tk.Button(root, text='Another Point', command=show).grid(row=1, column=0, padx=50, pady=20)
        tk.Button(root, text="Exit", command=Close).grid(row=2, column=0, padx=50, pady=10)
        root.mainloop()

    # === SINGLE-STEP PREDICTION ===
    elif steps == 1:
        WINDOW_SIZE = win1st
        X1, y1 = df_to_X_y(disp_df1, WINDOW_SIZE)
        X1 = X1.astype('float64')
        y1 = y1.astype('float64')

        tmdomX = np.repeat(tms_ar[:-1].reshape(1, -1, 1), repeats=X1.shape[0], axis=0)
        tmdomY = np.repeat(tms_ar[-1], repeats=y1.shape[0])
        X1 = np.stack((X1, tmdomX.squeeze()), axis=-1)
        y1 = np.stack((y1, tmdomY), axis=-1)

        vt = int(0.6 * len(displacements))
        ve = int(0.2 * len(displacements))
        X_train1, y_train1 = X1[:vt], y1[:vt]
        X_val1, y_val1 = X1[vt:vt + ve], y1[vt:vt + ve]
        X_test1, y_test1 = X1[vt + ve:vt + 2 * ve], y1[vt + ve:vt + 2 * ve]

        model1 = build_time_gated_lstm_model(input_shape=(WINDOW_SIZE, 2), units=75)
        model1.summary()
        history = model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=epochs, batch_size=128)
        acc = history.history['root_mean_squared_error']
        val_acc = history.history['val_root_mean_squared_error']
        test_predictions = model1.predict(X_test1)
        RMSE = round(math.sqrt(mean_squared_error(y_test1[:, 0], test_predictions[:, 0])), 3)

        MessageBox = ctypes.windll.user32.MessageBoxW
        MessageBox(None, f'The dataset has been trained and tested.\n The Root Mean Squared Error is {RMSE} \n', 'Window title', 0)

        if str(ischecked) == 'true':
            plotCurves(acc, val_acc, epochs, RMSE)

        dd_as_np = disp_df1.to_numpy()
        Xe = [[[a] for a in dd_as_np[1:, j]] for j in range(dd_as_np.shape[1])]
        Xe1 = [dd_as_np[0, j] for j in range(dd_as_np.shape[1])]
        Xe = np.array(Xe).astype('float64')
        Xe1 = np.array(Xe1).astype('float64')
        tmdomXFIN = np.repeat(tms_ar[1:].reshape(1, -1, 1), repeats=Xe.shape[0], axis=0)
        XeFIN = np.stack((Xe[:, :, 0], tmdomXFIN.squeeze()), axis=-1)

        test_predictFIN = model1.predict(XeFIN)

        root = tk.Tk()
        root.geometry("200x150")
        tk.Button(root, text='Another Point', command=show).grid(row=1, column=0, padx=50, pady=20)
        tk.Button(root, text="Exit", command=Close).grid(row=2, column=0, padx=50, pady=10)
        root.mainloop()

    else:
        arcpy.AddError("The number of predicted time steps is not valid.\nRefill the field of this parameter with a valid value.")
        sys.exit(0)
