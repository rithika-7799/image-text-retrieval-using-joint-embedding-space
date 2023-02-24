import pickle
import tensorflow as tf

from combined import get_combined_model
model = get_combined_model()
model.load_weights('artifacts/Combined_model_weights_0.1_2_0.1_0.1')
print(model)