import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


faulty_training_path = r'C:\Users\saumi\OneDrive\Desktop\control\faulty_training.csv'
fault_free_training_path = r'C:\Users\saumi\OneDrive\Desktop\control\fault_free_training.csv'
fault_free_testing_path = r'C:\Users\saumi\OneDrive\Desktop\control\fault_free_testing.csv'
faulty_testing_path = r'C:\Users\saumi\OneDrive\Desktop\control\faulty_testing.csv'


faulty_training = pd.read_csv(faulty_training_path)
fault_free_training = pd.read_csv(fault_free_training_path)
fault_free_testing = pd.read_csv(fault_free_testing_path)
faulty_testing = pd.read_csv(faulty_testing_path)


training_data = pd.concat([faulty_training, fault_free_training])


X_train = training_data.iloc[:, :-1].values  
y_train = training_data.iloc[:, -1].values 

X_test = pd.concat([fault_free_testing, faulty_testing]).iloc[:, :-1].values 
y_test = pd.concat([fault_free_testing, faulty_testing]).iloc[:, -1].values


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.transform(y_test.reshape(-1, 1))
