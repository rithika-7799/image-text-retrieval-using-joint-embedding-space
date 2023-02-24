import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json
import random
import json
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import pandas as pd
from image_preprocessing import extract_features
from pathlib import Path
Path("data/training/batches").mkdir(parents=True, exist_ok=True)


def text_preprocessing(corpus):
    
    # input: corpus
    # output: clean corpus without stop words, punctuations, and lower case
    #print("\nText preprocessing")
    clean = []
    stop_words = set(stopwords.words('english'))
    # tokenize sentence, remove stop words, punctuation and to lower case
    for sentence in corpus:
        tokens =nltk.word_tokenize(sentence)
        clean_sentence = [w.lower() for w in tokens if (not w in stop_words and w.isalpha())]
        clean.append(clean_sentence)

    num_instances = len(clean)
    #print("text cleaning complete, num_instances",num_instances)
    return clean

def vectorizing_data(corpus,MAX_SEQUENCE_LENGTH):
    
    """ vectorizing and splitting the data for training, testing, validating """
    # vectorizing the text samples and labels into a 2D integer tensor
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    print("saving the tokenizer")
    # saving
    with open('artifacts/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved")
    
    sequences = tokenizer.texts_to_sequences(corpus)
    dfseq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
    
    return dfseq

def extract_features_text(text):
    
    MAX_SEQUENCE_LENGTH = 30
    text=text_preprocessing(text)
    print("text preprocessing complete")
    features=vectorizing_data(text,MAX_SEQUENCE_LENGTH)
    print("text vectorizing complete")
    return np.array(features)


# placeholder function for image feature extraction
def extract_features_image(images, categories):
    # for image, category in enumerate(images,categories):
    #     print("data/final/images/"+category+"/"+image)
    #     print(extract_features("data/final/images/"+category+"/"+image))
    #     break

    return [extract_features("data/final/images/"+category+"/"+image) for image, category in zip(images,categories)]
    # return image

# to expand df and convert into features
# to expand df and convert into features
def batch_preprocessing(data_df):
    
    print("Batch preprocessing stage")
    # drop unnecessary columns
    data_df.drop(['id','category_id','coco_url','height','width','supercategory'],inplace=True,axis=1)
    
    print("\nConverting images to features")
    image_features= extract_features_image(data_df['file'].values.tolist(), data_df['category_name'].values.tolist()) # placeholder function
    image_features = [np.array(x) for x in image_features]
    data_df['image_features'] = image_features

    # expand dataframe captions
    data = []
    for index,row in data_df.iterrows():
        data.append([row['category_name'],row['file'],row['image_features'],row['caption_0']])
        data.append([row['category_name'],row['file'],row['image_features'],row['caption_1']])
        data.append([row['category_name'],row['file'],row['image_features'],row['caption_2']])
        data.append([row['category_name'],row['file'],row['image_features'],row['caption_3']])
        data.append([row['category_name'],row['file'],row['image_features'],row['caption_4']])

    data_df = pd.DataFrame(data, columns = ['category', 'image','image_features','caption'])
    
    # shuffle the data points
    print("shuffling dataframe")
    df_shuffled = data_df.sample(frac=1)
    
    # convert image and text to their feature vectors
    print("Converting text to features")
    text_features= extract_features_text(df_shuffled['caption'].values.tolist())
    text_features = [[x] for x in text_features]
    
    
    df_shuffled['text_features'] = text_features
    #df_shuffled['image_features'] = image_features
    
    print("Batch preprocessing complete\n")
    return df_shuffled

def create_batches(df,batch_size):
    
    '''
    input: dataframe and batch_size
    output: list containing each batch in the form of a dataframe
    
    '''
    batch_nums = round(len(df)/batch_size)
    
    dftemp = df.copy()
    batches = []
    batches_pickle = []
    for i in range(batch_nums):
        if(i%100==0):
            print("Creating batch ",i,"/",batch_nums)
        # randomly pick batch_size rows from dataset
        sample = dftemp.sample(n=batch_size,replace=False)
        # number of rows in each category
        countdf = sample.groupby(by=["category"]).count()['image']
        
        for cat,count in countdf.items():
            if count == 1:
                highest_cat = countdf.sort_values(ascending=False).head(1).index[0]
                remove_row  = sample[sample.category == highest_cat].sample(1)
                sample = sample.drop(remove_row.index)
                
                # add new row
                add_row  = df[df.category == cat].sample(1)
                sample = sample.append(add_row)
        assert(len(sample)==batch_size)  
        batches.append(sample)
        batches_pickle.append(sample)

        # when i reaches 50, save to pickle file and clear batches
        if(i!=0 and i%50==0):
            with open("artifacts/" + 'batch_'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(batches_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("saved batch ",i,"in pickle file")
                batches_pickle = []
    
        if(i==batch_nums-1):
            with open("artifacts/" + 'batch_'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(batches_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("saved batch ",i,"in pickle file")
                batches_pickle = []

        cat_mapping = {}
        with open("mapping_cat.pkl", "rb") as input_file:
            cat_map = pickle.load(input_file)
            for index, row in cat_map.iterrows():
                cat_mapping[row['x']]= row['y'] 
        sample['category'].replace(cat_mapping, inplace=True)
    print(len(batches),' batches created of size ',batch_size,' each.')
    return batches    
        
if __name__ == "__main__":
    with open('data/final/final_dict.json') as json_file:
        data = json.load(json_file)
    data_df = pd.DataFrame(data)
    df = batch_preprocessing(data_df)
    b = create_batches(df,batch_size=32)