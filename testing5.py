import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
from collections import deque


model = load_model(
    "C:/Users/saumi/OneDrive/Desktop/control/fault_prediction_model.h5",
    custom_objects={"MeanSquaredError": MeanSquaredError()}
)


def load_data():
    df_faulty = pd.read_csv("C:/Users/saumi/OneDrive/Desktop/control/faulty_testing.csv")
    df_fault_free = pd.read_csv("C:/Users/saumi/OneDrive/Desktop/control/fault_free_testing.csv")
    df = pd.concat([df_faulty, df_fault_free], ignore_index=True)
    return df

def preprocess_data(df):
    scaler = joblib.load('scaler.pkl') 
    features = df[['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 
                   'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12',
                   'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18',
                   'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24',
                   'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30',
                   'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36',
                   'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_1',
                   'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9',
                   'xmv_10', 'xmv_11']]
    features_normalized = scaler.transform(features)
    return features_normalized


df = load_data()
features_normalized = preprocess_data(df)
features_normalized = np.reshape(features_normalized, (features_normalized.shape[0], 1, features_normalized.shape[1]))
actual_data = deque(maxlen=100)
predictions = deque(maxlen=100)
plotting_data = deque(maxlen=100) 


def update(frame):
    global data_index
    if data_index >= len(features_normalized):
        return  
    

    prediction = model.predict(features_normalized[data_index].reshape(1, 1, -1))
    predicted_fault_number = prediction[0][0] 
    

    actual_fault_number = df['faultNumber'].values[data_index]  
    actual_data.append(actual_fault_number)
    predictions.append(predicted_fault_number)
    

    plotting_data.append(predicted_fault_number)
    
    
    print(f"Actual: {actual_fault_number}, Predicted: {predicted_fault_number}")
    
    data_index += 1
    
    ax1.clear()
    ax2.clear()
    

    ax1.scatter(range(len(actual_data)), list(predictions), color='red', label='Predicted')
    ax2.scatter(range(len(plotting_data)), list(plotting_data), color='red', label='Predicted')
    
    ax1.set_title('Actual Fault Numbers (as Predicted)')
    ax2.set_title('Predicted Fault Numbers')
    ax1.legend()
    ax2.legend()
    ax1.set_ylim(min(predictions) - 1, max(predictions) + 1)
    ax2.set_ylim(min(plotting_data) - 1, max(plotting_data) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
data_index = 0

ani = FuncAnimation(fig, update, interval=200) 

plt.show()
