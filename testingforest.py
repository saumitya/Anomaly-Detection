import pandas as pd
import pickle

model_filename = r'C:\Users\saumi\OneDrive\Desktop\control\isolation_forest_model.pkl'
with open(model_filename, 'rb') as file:
    iso_forest = pickle.load(file)

faulty_test_data_path = r'C:\Users\saumi\OneDrive\Desktop\control\normalized_faulty_testing.csv'
faultfree_test_data_path = r'C:\Users\saumi\OneDrive\Desktop\control\normalized_fault_free_testing.csv'

df_faulty_test = pd.read_csv(faulty_test_data_path)
df_faultfree_test = pd.read_csv(faultfree_test_data_path)

df_test = pd.concat([df_faulty_test, df_faultfree_test], axis=0)

X_test = df_test.drop(columns=['faultNumber'])


y_pred = iso_forest.predict(X_test)

df_test['prediction'] = y_pred


output_filename = r'C:\Users\saumi\OneDrive\Desktop\control\isolation_forest_predictions.csv'
df_test.to_csv(output_filename, index=False)

print(f'Predictions saved to {output_filename}')
