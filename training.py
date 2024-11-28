import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 


normalized_faulty_training_path = "C:/Users/saumi/OneDrive/Desktop/control/normalized_faulty_training.csv"
normalized_fault_free_training_path = "C:/Users/saumi/OneDrive/Desktop/control/normalized_fault_free_training.csv"

normalized_faulty_training = pd.read_csv(normalized_faulty_training_path)
normalized_fault_free_training = pd.read_csv(normalized_fault_free_training_path)

training_data = pd.concat([normalized_faulty_training, normalized_fault_free_training], ignore_index=True)

features_columns = ['xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10',
                    'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20',
                    'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30',
                    'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40',
                    'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11']
X = training_data[features_columns]  
y = training_data['faultNumber']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl') 

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1)) 
model.compile(optimizer='adam', loss=MeanSquaredError()) 


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))


model.save("C:/Users/saumi/OneDrive/Desktop/control/fault_prediction_model.h5")
