import numpy as np
import pandas as pd
from keras.models import load_model
from keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler


model = load_model(
    "C:/Users/saumi/OneDrive/Desktop/control/fault_prediction_model.h5",
    custom_objects={"MeanSquaredError": MeanSquaredError()}
)


normalized_fault_free_testing_path = "C:/Users/saumi/OneDrive/Desktop/control/normalized_fault_free_testing.csv"
normalized_faulty_testing_path = "C:/Users/saumi/OneDrive/Desktop/control/normalized_faulty_testing.csv"

normalized_fault_free_testing = pd.read_csv(normalized_fault_free_testing_path)
normalized_faulty_testing = pd.read_csv(normalized_faulty_testing_path)


test_data = pd.concat([normalized_fault_free_testing, normalized_faulty_testing], ignore_index=True)


features_columns = ['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10', 'xmeas_11',
                    'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20', 'xmeas_21', 
                    'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 
                    'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40', 'xmeas_41', 
                    'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11']
features = test_data[features_columns].values
actual_values = test_data['faultNumber'].values


scaler = MinMaxScaler()
features = scaler.fit_transform(features)


features = np.reshape(features, (features.shape[0], 1, features.shape[1]))


actual_data = []
predictions = []


data_index = 0


def update(frame):
    global data_index
    if data_index >= len(features):
        return  
    
   
    prediction = model.predict(features[data_index].reshape(1, 1, -1))
    
   
    actual_value = actual_values[data_index]
    actual_data.append(actual_value)
    predictions.append(prediction[0][0])
    
   
    print(f"Data index: {data_index}")
    print(f"Actual value: {actual_value}")
    print(f"Predicted value: {prediction[0][0]}")
    
   
    data_index += 1
    
    ax1.clear()
    ax2.clear()
    
    
    ax1.plot(actual_data, color='blue', label='Actual faultNumber')
    ax2.plot(predictions, color='red', label='Predicted faultNumber')
    
   
    ax1.set_title('Actual Values')
    ax2.set_title('Predicted Values')
    ax1.legend()
    ax2.legend()
    ax1.set_ylim(min(actual_data) - 1, max(actual_data) + 1)
    ax2.set_ylim(min(predictions) - 1, max(predictions) + 1)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))


ani = FuncAnimation(fig, update, interval=200)  

plt.show()
