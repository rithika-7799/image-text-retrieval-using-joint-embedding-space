from image_preprocessing import extract_features
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
import numpy as np
import pickle
from tqdm import tqdm
from batch_creation import  extract_features_text
import json
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
from combined import get_combined_model
from batch_creation import text_preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from image_preprocessing import extract_features

def _pairwise_distances(embeddings, embeddings2, squared=False):
    embeddings = tf.cast(embeddings, tf.float32)#embeddings.astype('float32')
    embeddings2 = tf.cast(embeddings2, tf.float32)#embeddings2.astype('float32')
    # print(embeddings.shape)
    # print(embeddings2.shape)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings2))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.compat.v1.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances[0]

def get_text_features(search_key):
    with open('artifacts/tokenizer.pickle', 'rb') as handle:

        tokenizer = pickle.load(handle)

    MAX_SEQUENCE_LENGTH=30

    text = text_preprocessing([search_key])

    sequences = tokenizer.texts_to_sequences(text)

    text_features = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
    print(search_key)
    #text_features= extract_features_text(search_key)
    return text_features

def image_text_embedding_extraction(data_df_2): 
    img_list=[]
    image_embed= []
    for i in range(len(data_df_2)):
          img1=data_df_2['image_embedding'].iloc[i]
          img_list.append(img1)
    image_embed=np.array(img_list)      
    image_embed_1=np.reshape(image_embed,(len(data_df_2),64))
    text_list=[]
    text_embed=[]
    '''
    text_features= get_text_features(search_key)
    for i in range(len(data_df_2)):
        
          text_list.append(text_features)
    text_embed=np.array(text_list)      
    text_embed_1=np.reshape(text_embed,(len(data_df_2),30))     
    '''

    return image_embed_1

def text_reshape(data_df_2,text_features): 
    text_list=[]
    text_embed=[]
   
    for i in range(len(data_df_2)):
        
          text_list.append(text_features)
    text_embed=np.array(text_list)      
    text_embed_1=np.reshape(text_embed,(len(data_df_2),64))     

    return text_embed_1

## Read the saved model
def search_key_neighbors(search_key,img_path,is_image=False):
    batch_size= 1
    model = get_combined_model()
    with open('artifacts/multiple_batch_image_embedding.pickle', 'rb') as handle:
        data_df_2 = pickle.load(handle)
    # data_df_2 = None  #load the pickle file for image embedding  here
    data_df_1 = None
    if is_image==False:
        
        image_embed= tf.fill((batch_size,7 * 7 * 2048), 1, name=None)
        text_embed_1=get_text_features(search_key)
       
        comb_pred=model.predict([image_embed,text_embed_1])  #getting the prediction from the model
        image_embed_1= image_text_embedding_extraction(data_df_2) 
        text_embed_r=text_reshape(data_df_2,comb_pred[:,64:])
        pd=_pairwise_distances(image_embed_1 ,text_embed_r)

        pd=pd.numpy()
        idx=np.argpartition(pd,5)
        near_df=data_df_2.iloc[idx[0:5],]
        # print(near_df)
        # for i in range(len(near_df)):
            
        #     plt.figure(figsize=(5, 5))
        #     #ax = plt.subplot(4, 8, i + 1)
        #     input_shape = (224, 224, 3)
        #     img_path=near_df['path'].iloc[i]
        #     img=tf.keras.utils.load_img(img_path, target_size=(
        #             input_shape[0], input_shape[1]))
        #     #print(i)
        #     plt.imshow(img)
    else:
        text_embed= tf.fill((batch_size* 30), 1, name=None)
        image_embed=extract_features(img_path)
        #data_df_2   #load the pickle file for image embedding  here
        comb_pred=model.predict([image_embed,text_embed])  #getting the prediction from the model
        text_embed_1= image_text_embedding_extraction(data_df_2) 
        image_embed_r=text_reshape(data_df_2,comb_pred[:,64:])
        pd=_pairwise_distances(image_embed_r ,text_embed_1)

        pd=pd.numpy()
        idx=np.argpartition(pd,5)
        near_df=data_df_1.iloc[idx[0:5],]
    
    return near_df
    '''
    #if reading f
    data_df_1[["coco_url","path","image_features"]]  # Read the
    image_embed_1,text_embed_1=image_embedding_extraction(data_df_2)
    '''

search_key="bag"
search_key_neighbors(search_key,img_path="0",is_image=False)