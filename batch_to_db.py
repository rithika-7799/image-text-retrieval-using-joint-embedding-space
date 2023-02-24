from text_model import get_text_model_saved
from image_model import get_image_model_saved
from combined import get_combined_model
from batch_creation import text_preprocessing
import numpy as np
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
import tensorflow as tf

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
from batch_creation import create_batches,batch_preprocessing
import pandas as pd
import os
from tqdm import tqdm
#import get_saved_model

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt

MAX_SEQUENCE_LENGTH = 30

from text_model import get_text_model_saved
from image_model import get_image_model_saved

# get combined model
combined_model = get_combined_model()

def _convert_(text_features,image_features):
    batch_size = None
    if text_features.empty and image_features.empty:
        return []
    elif not text_features.empty and image_features.empty:
        batch_size = text_features.size
    elif text_features.empty and not image_features.empty:
        batch_size = image_features.size
    else:
        batch_size = min(len(text_features), len(image_features))
    if not text_features.empty:
        # convert text to features
        # load tokenizer and vectorize
        # with open('artifacts/tokenizer.pickle', 'rb') as handle:
        #     tokenizer = pickle.load(handle)
        
        # text = text_preprocessing(text)
        # sequences = tokenizer.texts_to_sequences(text)
        # text_features = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
        b_in = tf.convert_to_tensor(np.array(text_features.values.tolist()[:32])) #tf.fill((1, 30), 1, name=None)
        if(len(b_in.shape)==3):
            b_in = tf.reshape(b_in, [b_in.shape[0] , b_in.shape[1] * b_in.shape[2]])
    else:
        b_in = tf.fill((batch_size, 30), 1, name=None)

    if not image_features.empty:
        a_in = tf.convert_to_tensor(np.array(image_features.values.tolist()[:32]))
    else:
        a_in = tf.fill((batch_size,7 * 7 * 2048), 1, name=None)
        
    prediction = combined_model.predict([a_in,b_in])
    
    # print('prediction.shape',prediction.shape)

    image_pred = prediction[:,:64]
    text_pred = prediction[:,64:]
    return image_pred, text_pred
    
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

if __name__=='__main__':

    print("inside main")
    root_dir = 'artifacts/'
    filenames = sorted(get_file_list(root_dir))
    batch_files = [f for f in filenames if "batch_" in f and "multiple_batch" not in f]
    # batch_files = ["batch_50", "batch_100", "batch_150"]
    multiple_batch_text_embedding = pd.DataFrame(columns=['category', 'image', 'text_embedding'])
    multiple_batch_image_embedding = pd.DataFrame(columns=['category', 'image', 'image_embedding'])
    for file_name in batch_files:
        batch_no = file_name.split("_")[-1]
        
        with open(file_name, 'rb') as handle:
            batches=pickle.load(handle)
        print(f"Total batches for {file_name}: {len(batches)}")
        for batch in batches:
            text_features = batch['text_features']
            image_features = batch['image_features']
            image_emb, text_emb = _convert_(text_features, image_features)
            batch_image_df = batch.copy()
            batch_text_df = batch.copy()

            batch_image_df.drop(["image_features", "caption", "text_features"], inplace=True, axis = 1)
            batch_image_df["image_embedding"] = pd.NaT
            batch_image_df = batch_image_df.astype(dtype= {"image_embedding":"object"})
            count = 0
            for i in batch_image_df.index:
                # print(batch_image_df.at[i, 'image_embedding'])
                try:
                    batch_image_df.at[i, 'image_embedding'] = [image_emb[count, :]]
                except Exception as e:
                    print(e)
                count+=1
            
            batch_text_df.drop(["image_features", "text_features"], inplace=True, axis = 1)
            batch_text_df["text_embedding"] = pd.NaT
            batch_text_df = batch_text_df.astype(dtype= {"text_embedding":"object"})
            count = 0
            for i in batch_text_df.index:
                try:
                    batch_text_df.at[i, 'text_embedding'] = [text_emb[count, :]]
                except Exception as e:
                    print(e)
                count+=1
            multiple_batch_image_embedding = multiple_batch_image_embedding.append(batch_image_df)
            multiple_batch_text_embedding = multiple_batch_text_embedding.append(batch_text_df)
            # break
    with open("artifacts/" + 'multiple_batch_text_embedding.pickle', 'wb') as handle:
        pickle.dump(multiple_batch_text_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved embedding batch ","in pickle file")
    
    with open("artifacts/" + 'multiple_batch_image_embedding.pickle', 'wb') as handle:
        pickle.dump(multiple_batch_image_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved embedding batch ","in pickle file")