import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler


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
    scaler = StandardScaler()
    features = df[['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 
                   'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12',
                   'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18',
                   'xmeas_19', 'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24',
                   'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30',
                   'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36',
                   'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_1',
                   'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9',
                   'xmv_10', 'xmv_11']]
    features_normalized = scaler.fit_transform(features)
    return features_normalized

def update(frame):
    global actual_data, predictions
    df = load_data()
    features_normalized = preprocess_data(df)
    
    
    features_normalized = np.reshape(features_normalized, (features_normalized.shape[0], 1, features_normalized.shape[1]))
    
    
    prediction = model.predict(features_normalized[-1].reshape(1, 1, -1)) 
    
    
    actual_data.append(df['faultNumber'].values[-1])  
    predictions.append(prediction[0][0]) 
    
    
    if len(actual_data) > window_size:
        actual_data.pop(0)
        predictions.pop(0)
    
    ax1.clear()
    ax2.clear()
    
    
    ax1.plot(actual_data, color='blue', label='Actual')
    ax2.plot(predictions, color='red', label='Predicted')
    

    ax1.set_title('Actual Values')
    ax2.set_title('Predicted Values')
    ax1.legend()
    ax2.legend()
    ax1.set_ylim(min(actual_data) - 1, max(actual_data) + 1)
    ax2.set_ylim(min(predictions) - 1, max(predictions) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
window_size = 100 

actual_data = []
predictions = []


ani = FuncAnimation(fig, update, interval=0)  

plt.show()