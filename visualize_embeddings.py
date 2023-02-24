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

from tqdm import tqdm
#import get_saved_model

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt

MAX_SEQUENCE_LENGTH = 30

# a function that takes raw text(any text) and image, and extracts the PCA/TSNE reprentation of their embeddings.

from text_model import get_text_model_saved
from image_model import get_image_model_saved


def extract_PCA_TSNE(text,image,labels):
    
    # convert image to features
    
    # convert text to features
    # load tokenizer and vectorize
    with open('artifacts/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    text = text_preprocessing(text)
    sequences = tokenizer.texts_to_sequences(text)
    text_features = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
    
    # get combined model
    batch_size = len(text)
    a_in = tf.fill((batch_size,7 * 7 * 2048), 1, name=None)
    b_in = tf.convert_to_tensor(np.array(text_features)) #tf.fill((1, 30), 1, name=None)
    #print('a_in.shape',a_in.shape)
    #print('b_in.shape',b_in.shape)
    combined_model = get_combined_model()
    combined_model.load_weights('Combined_model_weights_0.15_2_1_0.5')

    prediction = combined_model.predict([a_in,b_in])
    
    print('prediction.shape',prediction.shape)

    image_pred = prediction[:,:64]
    text_pred = prediction[:,64:]
    
    print('text_pred.shape',text_pred.shape)
    
    time_start = time.time()
    tsne_text = TSNE(n_components=2, verbose=1, metric='euclidean',n_iter=1000)
    tsne_pca_text = tsne_text.fit_transform(text_pred)
    print('t-SNE done for text! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    # Plot a scatter plot from the generated t-SNE results
    colormap = plt.cm.get_cmap('coolwarm')
    scatter_plot = plt.scatter(tsne_pca_text[:,0],tsne_pca_text[:,1],c=labels, cmap=colormap)
    plt.colorbar(scatter_plot)
    plt.show()
    

if __name__=='__main__':

    print("inside main")
    with open('batch_0.pickle', 'rb') as handle:
        batches=pickle.load(handle)

    #with open('mapping_cat.pkl', 'rb') as handle:
    #    batch=pickle.load(handle)[0]
    d = {'bench': [0], 'car': [1], 'dog': [2], 'elephant': [3], 'handbag': [4], 'horse': [5], 'person': [6], 'traffic light': [7], 'truck': [8], 'umbrella': [9], 'zebra': [10]}
    
    text = batches[0]['caption']
    labels = batches[0]['category']
    l = []
    for label in labels:
        l.append(d[label])


    #text=["elephant near tree","elephant on the grass","person in the car"]
    image = [1,1,2]
    #labels = [1,1,2]
    extract_PCA_TSNE(text,image,l)