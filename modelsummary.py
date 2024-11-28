from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

model = load_model(
    "C:/Users/saumi/OneDrive/Desktop/control/fault_prediction_model.h5",
    custom_objects={"MeanSquaredError": MeanSquaredError()}
)

model.summary()
