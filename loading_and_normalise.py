import pandas as pd
from sklearn.preprocessing import StandardScaler


faulty_training_path = "C:/Users/saumi/OneDrive/Desktop/control/faulty_training.csv"
fault_free_training_path = "C:/Users/saumi/OneDrive/Desktop/control/fault_free_training.csv"
fault_free_testing_path = "C:/Users/saumi/OneDrive/Desktop/control/fault_free_testing.csv"
faulty_testing_path = "C:/Users/saumi/OneDrive/Desktop/control/faulty_testing.csv"

faulty_training = pd.read_csv(faulty_training_path)
fault_free_training = pd.read_csv(fault_free_training_path)
fault_free_testing = pd.read_csv(fault_free_testing_path)
faulty_testing = pd.read_csv(faulty_testing_path)


all_data = pd.concat([faulty_training, fault_free_training, fault_free_testing, faulty_testing], ignore_index=True)

scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data.drop(columns=['faultNumber', 'simulationRun', 'sample']))


normalized_data = pd.DataFrame(all_data_scaled, columns=all_data.columns[3:])
normalized_data['faultNumber'] = all_data['faultNumber']
normalized_data['simulationRun'] = all_data['simulationRun']
normalized_data['sample'] = all_data['sample']


normalized_faulty_training = normalized_data.iloc[:len(faulty_training)]
normalized_fault_free_training = normalized_data.iloc[len(faulty_training):len(faulty_training) + len(fault_free_training)]
normalized_fault_free_testing = normalized_data.iloc[len(faulty_training) + len(fault_free_training):len(faulty_training) + len(fault_free_training) + len(fault_free_testing)]
normalized_faulty_testing = normalized_data.iloc[len(faulty_training) + len(fault_free_training) + len(fault_free_testing):]


normalized_faulty_training.to_csv("C:/Users/saumi/OneDrive/Desktop/control/normalized_faulty_training.csv", index=False)
normalized_fault_free_training.to_csv("C:/Users/saumi/OneDrive/Desktop/control/normalized_fault_free_training.csv", index=False)
normalized_fault_free_testing.to_csv("C:/Users/saumi/OneDrive/Desktop/control/normalized_fault_free_testing.csv", index=False)
normalized_faulty_testing.to_csv("C:/Users/saumi/OneDrive/Desktop/control/normalized_faulty_testing.csv", index=False)
