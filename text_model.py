import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, Input, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Concatenate, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
import os
import pandas as pd
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import array
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
LABEL_COUNT= 0

def load_embeddings():

    # load glove embeddings and store in dict word2vec
    word2vec = {}

    with open(os.path.join('glove.6B.%sd.txt' % EMBEDDING_DIM),encoding="utf8") as f:
      # is just a space-separated text file in the format:
      # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print('Found %s word vectors.' % len(word2vec))
    return word2vec

def prepare_embedding_matrix(word_index):
    """ preparing embedding matrix with our data set """

    embeddings_index = load_embeddings()
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, num_words

def vectorizing_data(corpus,labels,MAX_SEQUENCE_LENGTH,split=True,):
    
    """ vectorizing and splitting the data for training, testing, validating """
    # vectorizing the text samples and labels into a 2D integer tensor
    
    #label_s = df['label'].tolist()
    
    if(split==True):
        label_s = labels
        
        # dont change max_sequence_length, causes errors cause it varies
        #word_count = lambda sentence: len(sentence)
        #longest_sentence = max(corpus, key=word_count)
        #MAX_SEQUENCE_LENGTH = len(longest_sentence)


        l = list(set(label_s))
        l.sort()
        labels_index = dict([(j,i) for i, j in enumerate(l)]) 
        labels = [labels_index[i] for i in label_s]

        print('Found %s texts.' % len(corpus))
        print('labels_index --- ', labels_index)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)
        sequences = tokenizer.texts_to_sequences(corpus)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        """
        if not home_path + 'word_index_tutorial.pickle' in os.listdir():
            with open(home_path + 'word_index_tutorial.pickle', 'wb') as handle:
                pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """

        dfseq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
        labels = to_categorical(np.asarray(labels))
        print('Shape of df tensor:', dfseq.shape)
        print('Shape of label tensor:', labels.shape)

        # randomizing and splitting the df into a training set, test set and a validation set
        indices = np.arange(dfseq.shape[0])
        np.random.shuffle(indices)
        dfseq= dfseq[indices]
        labels = labels[indices]
        num_validation_samples = int(VALIDATION_SPLIT * dfseq.shape[0])

        x_train = dfseq[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = dfseq[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]
        x_test = x_train[-num_validation_samples:]
        y_test = y_train[-num_validation_samples:]
        return x_train, y_train, x_test, y_test, x_val, y_val, word_index
    
    else: # dont split into train test, only vectorizing for triplet loss
        label_s = labels
        
        # dont change max_sequence_length, causes errors cause it varies
        #word_count = lambda sentence: len(sentence)
        #longest_sentence = max(corpus, key=word_count)
        #MAX_SEQUENCE_LENGTH = len(longest_sentence)

        l = list(set(label_s))
        l.sort()
        labels_index = dict([(j,i) for i, j in enumerate(l)]) 
        labels = [labels_index[i] for i in label_s]

        print('Found %s texts.' % len(corpus))
        print('labels_index --- ', labels_index)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)
        sequences = tokenizer.texts_to_sequences(corpus)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        dfseq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
        labels = to_categorical(np.asarray(labels))
        print('Shape of df tensor:', dfseq.shape)
        print('Shape of label tensor:', labels.shape)

        return dfseq,labels,word_index
    
def generate_training_data(df,cap1):
    
    # input: dataframe
    # output: corpus, classes onehot encoded 
    
    if(cap1==False):
        print("Generating data for all 5 annotations")
        print("length of dataframe: ",len(df))
        corpus = []
        corpus.extend(df['caption_0'].values)
        corpus.extend(df['caption_1'].values)
        corpus.extend(df['caption_2'].values)
        corpus.extend(df['caption_3'].values)
        corpus.extend(df['caption_4'].values)
        print("length of corpus: ",len(corpus))

        labels = (df['category_name'].values.tolist())*5
        LABEL_COUNT = len(set(labels))


        df['category_name'] = pd.factorize(df.category_name)[0]
        classes = array(df['category_name'].values.tolist())

        classes = [str(x) for x in classes]
        classes = classes*5
        classes = array([[int(x)] for x in classes])
        factorized_labels = classes

        # one hot encoding
        encoder = OneHotEncoder(sparse=False)
        # transform data
        onehot_classes = encoder.fit_transform(classes)
        print("number of target classes: ",len(onehot_classes))
        return (corpus,onehot_classes,labels,factorized_labels)
    
    else:
        
        print("generating data only for 1st annotation")
        print("length of dataframe: ",len(df))
        corpus = []
        corpus.extend(df['caption_0'].values)
        print("length of corpus: ",len(corpus))

        labels = (df['category_name'].values.tolist())
        LABEL_COUNT = len(set(labels))

        df['category_name'] = pd.factorize(df.category_name)[0]
        classes = array(df['category_name'].values.tolist())

        classes = [str(x) for x in classes]
        classes = array([[int(x)] for x in classes])
        factorized_labels = classes

        # one hot encoding
        encoder = OneHotEncoder(sparse=False)
        # transform data
        onehot_classes = encoder.fit_transform(classes)
        print("number of target classes: ",len(onehot_classes))
        return (corpus,onehot_classes,labels,factorized_labels)
    

    
def text_preprocessing(corpus):
    
    # input: corpus
    # output: clean corpus without stop words, punctuations, and lower case

    clean = []
    stop_words = set(stopwords.words('english'))

    # tokenize sentence, remove stop words, punctuation and to lower case
    for sentence in corpus:
        tokens =nltk.word_tokenize(sentence)
        clean_sentence = [w.lower() for w in tokens if (not w in stop_words and w.isalpha())]
        clean.append(clean_sentence)

    num_instances = len(clean)
    print("text cleaning complete, num_instances",num_instances)
    return clean

def text_model(embedding_matrix, num_words):
    
    model=Sequential()
    embedding=Embedding(num_words+1, # number of unique tokens
                        EMBEDDING_DIM, #number of features
                        embeddings_initializer=Constant(embedding_matrix), # initialize 
                        input_length=MAX_SEQUENCE_LENGTH, 
                        trainable=False)

    model.add(embedding)
    model.add(Dropout(0.2))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.5))
    #model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    model.layers[4]._name = 'text_128embed_layer'
    model.layers[0]._name = 'input_text'
    return model

with open('data/final/final_dict.json') as json_file:
    data = json.load(json_file)


def get_text_model():
    data_df = pd.DataFrame(data)
    # data_df.tail(5)

    sss = ShuffleSplit(n_splits=1, test_size=0.85)

    data_size = len(data_df)
    X = np.reshape(np.random.rand(data_size*2),(data_size,2))
    y = np.random.randint(2, size=data_size)

    sss.get_n_splits(X, y)
    train_index, test_index = next(sss.split(X, y)) 

    Filter_df  = data_df[data_df.index.isin(train_index)]
    corpus, targets, labels,factorized_labels = generate_training_data(Filter_df,cap1=False)
    corpus = text_preprocessing(corpus)

    print('\nvectorizing data')
    x_train, y_train, x_test, y_test, x_val, y_val, word_index = vectorizing_data(corpus,labels,MAX_SEQUENCE_LENGTH,split=True)

    embedding_matrix, num_words = prepare_embedding_matrix(word_index)
    return text_model(embedding_matrix, num_words)
    # text_model(embedding_matrix, num_words).summary()


def get_text_model_saved():
    model =  load_model('artifacts/text_model_1.h5')
    print("Loaded Text Model")
    return model