import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time

model_filename = r'C:\Users\saumi\OneDrive\Desktop\control\isolation_forest_model.pkl'
with open(model_filename, 'rb') as file:
    iso_forest = pickle.load(file)

faulty_test_data_path = r'C:\Users\saumi\OneDrive\Desktop\control\normalized_faulty_testing.csv'
faultfree_test_data_path = r'C:\Users\saumi\OneDrive\Desktop\control\normalized_fault_free_testing.csv'

df_faulty_test = pd.read_csv(faulty_test_data_path)
df_faultfree_test = pd.read_csv(faultfree_test_data_path)

df_test = pd.concat([df_faulty_test, df_faultfree_test], axis=0)

X_test = df_test.drop(columns=['faultNumber'])
actual_faults = df_test['faultNumber'].values

plt.ion()  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.set_title('Real-Time Fault Predictions')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Prediction')
scatter_pred, = ax1.plot([], [], 'bo', label='Prediction')  

ax2.set_title('Real-Time Actual Fault Data')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Actual Fault Number')
scatter_actual, = ax2.plot([], [], 'ro', label='Actual')  

correct_predictions = 0 


def update_plots(i, predictions, actuals):
    global correct_predictions
    
  
    scatter_pred.set_xdata(np.append(scatter_pred.get_xdata(), i))
    scatter_pred.set_ydata(np.append(scatter_pred.get_ydata(), predictions[i]))
    
    scatter_actual.set_xdata(np.append(scatter_actual.get_xdata(), i))
    scatter_actual.set_ydata(np.append(scatter_actual.get_ydata(), actuals[i]))
    

    if predictions[i] == actuals[i]:
        correct_predictions += 1
    

    accuracy = (correct_predictions / (i + 1)) * 100
    
   
    ax1.set_title(f'Real-Time Fault Predictions (Accuracy: {accuracy:.2f}%)')
    ax2.set_title('Real-Time Actual Fault Data')
    
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    plt.draw()
    plt.pause(0.001)


predictions = iso_forest.predict(X_test)

for i in range(len(X_test)):
    update_plots(i, predictions, actual_faults)
    time.sleep(1)  

plt.ioff()
plt.show()
