import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
def get_image_model():
    model = tf.keras.Sequential([
     tf.keras.layers.Dense(1024, input_shape=(7 * 7 * 2048,), activation="relu"),
     tf.keras.layers.Dense(512, activation="relu"),
     tf.keras.layers.Dense(256, activation="relu"),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.Dense(64, activation="relu",),
     tf.keras.layers.Dense(11, activation="softmax")
    ])
    model.layers[3]._name = 'image_128embed_layer'
    return model

def get_image_model_saved(): 
    model = load_model('artifacts/image_model_1.h5')
    print('Loaded Image Model')
    return model