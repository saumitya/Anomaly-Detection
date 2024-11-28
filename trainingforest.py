import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle


faulty_train_data_path = r'C:\Users\saumi\OneDrive\Desktop\control\normalized_faulty_training.csv'
faultfree_train_data_path = r'C:\Users\saumi\OneDrive\Desktop\control\normalized_fault_free_training.csv'

df_faulty_train = pd.read_csv(faulty_train_data_path)
df_faultfree_train = pd.read_csv(faultfree_train_data_path)


df_train = pd.concat([df_faulty_train, df_faultfree_train], axis=0)


X_train = df_train.drop(columns=['faultNumber'])


iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

iso_forest.fit(X_train)

model_filename = 'isolation_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(iso_forest, file)

print(f'Model trained on both faulty and fault-free data and saved as {model_filename}')
