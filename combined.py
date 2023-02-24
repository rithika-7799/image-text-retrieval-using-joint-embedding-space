import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from text_model import get_text_model_saved
from image_model import get_image_model_saved
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from loss_function_hard_negative import loss_function_positive_semi_hard_neg
from tensorflow.keras.optimizers import Adam
import json
# from batch_creation import create_batches,batch_preprocessing
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

extensions = ['.pickle']
def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list


def get_combined_model():
    text_model = get_text_model_saved()
    plot_model(text_model, to_file='images/text_model.png', show_shapes=True)
    layer_name = 'text_128embed_layer'
    text_model_intermediate= Model(inputs=text_model.input, outputs=text_model.get_layer(layer_name).output)


    for layer in text_model_intermediate.layers[:-1]:
        layer._name = "image"+ layer._name
        layer.trainable = False
    # text_model_intermediate.summary()
    # for layer in text_model_intermediate.layers:
    #     print(layer.name, layer.trainable)

    text_model_embedding = Sequential()
    text_model_embedding.add(text_model_intermediate)
    text_model_embedding.add(BatchNormalization())
    text_model_embedding.add(Dense(64, input_shape=(128,), activation='relu'))
    plot_model(text_model_embedding, to_file='images/text_model_embedding.png', show_shapes=True)

    # print(text_model_embedding.output_shape)


    image_model = get_image_model_saved()
    plot_model(image_model, to_file='images/image_model.png', show_shapes=True)
    layer_name = 'image_128embed_layer'
    image_model_intermediate= Model(inputs=image_model.input, outputs=image_model.get_layer(layer_name).output)


    for layer in image_model_intermediate.layers[:-1]:
        layer._name = "text"+ layer._name
        layer.trainable = False
    # image_model_intermediate.summary()
    # for layer in image_model_intermediate.layers:
    #     print(layer.name, layer.trainable)

    image_model_embedding = Sequential()
    image_model_embedding.add(image_model_intermediate)
    image_model_embedding.add(BatchNormalization())
    image_model_embedding.add(Dense(64, input_shape=(128,), activation='relu'))
    plot_model(image_model_embedding, to_file='images/image_model_embedding.png', show_shapes=True)

    # image_model_embedding.summary()

    # print(image_model_embedding.output_shape)

    # print(text_model_intermediate.input_shape, text_model_intermediate.output_shape )
    # print(image_model_intermediate.input_shape, image_model_intermediate.output_shape )
    concatted = Concatenate()([image_model_embedding.output, text_model_embedding.output ])
    combined_model = Model(inputs=[image_model_embedding.input, text_model_embedding.input], outputs=concatted)
    return combined_model

if __name__ == "__main__":
    combined_model = get_combined_model()
    combined_model.summary()
    # tb_callback = tf.keras.callbacks.TensorBoard("logdir/")
    # tb_callback.set_model(combined_model)
    # plot_model(combined_model, to_file='images/model.png', show_shapes=True)

    # with open('data/final/final_dict.json') as json_file:
    #     data = json.load(json_file)

    # data_df = pd.DataFrame(data)
    # df = batch_preprocessing(data_df)
    # b = create_batches(df,batch_size=32)

    # output = open('batches.pkl', 'wb')
    # pickle.dump(b, output)
    # output.close()

    # pkl_file = open('batches.pkl', 'rb')
    # b = pickle.load(pkl_file)
    # pkl_file.close()

    # a_in = tf.convert_to_tensor(np.array(b[1]['image_features'].values.tolist()[:32])) #tf.fill((1,7 * 7 * 2048), 1, name=None)
    # b_in = tf.convert_to_tensor(np.array(b[1]['text_features'].values.tolist()[:32])) #tf.fill((1, 30), 1, name=None)
    # # if shape is (32,1,30) to convert to (32,30)
    # if(len(b_in.shape)==3):
    #     b_in = tf.reshape(b_in, [b_in.shape[0] , b_in.shape[1] * b_in.shape[2]])
    
    # y_train = tf.convert_to_tensor(np.array(b[1]['category'].values.tolist()[:32])) #tf.fill((1, 10), 0, name=None)

    # combined_model.fit([a_in, b_in], y_train, epochs = 3, batch_size = 32)
    root_dir = 'artifacts/'
    filenames = sorted(get_file_list(root_dir))
    filenames = [f for f in filenames if "batch_" in f and "multiple_batch" not in f]
    optimizer = Adam(learning_rate=1e-3)
    epochs = 1
    losses = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for i in range(len(filenames)):
            print(filenames[i])
            pkl_file = open(filenames[i], 'rb')
            b = pickle.load(pkl_file)
            pkl_file.close()

    # Iterate over the batches of the dataset.
            for step, batch in enumerate(b):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    a_in = tf.convert_to_tensor(np.array(batch['image_features'].values.tolist()[:32]))
                    b_in = tf.convert_to_tensor(np.array(batch['text_features'].values.tolist()[:32]))
                    # if shape is (32,1,30) to convert to (32,30)
                    if(len(b_in.shape)==3):
                        b_in = tf.reshape(b_in, [b_in.shape[0] , b_in.shape[1] * b_in.shape[2]])
                    logits = combined_model([a_in,b_in], training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_function_positive_semi_hard_neg(tf.convert_to_tensor(np.array(batch['category'].values.tolist()[:32])), logits)
                    # print()

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, combined_model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, combined_model.trainable_weights))

                # Log every 200 batches.
                if step % 10 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(tf.reduce_mean(loss_value, 0).numpy()))
                    )
                    losses.append(float(tf.reduce_mean(loss_value, 0).numpy()))
                    print("Seen so far: %s samples" % ((step + 1) * 32))
                    
    
    
    with open("hyperparams.txt") as f:
        params = f.read()
        params.replace(",", "_")
        combined_model.save_weights("artifacts/models/combined_model_weights_"+params)
        textfile = open("artifacts/losses/combined_model_weights_"+params, "w")
        for element in losses:
            textfile.write(str(element) + "\n")
        textfile.close()