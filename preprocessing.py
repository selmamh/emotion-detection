import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from textblob import Word
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
# nltk.download('stopwords')
# nltk.download('wordnet')

def load_data(file):
  df = pd.read_csv(file)
  return df


def drop_value_from_target(df,emotion):
  df= df.loc[df["emotion"] != emotion]
  return df

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


def replace_contractions(text):
    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    contractions, contractions_re = _get_contractions(contraction_dict)
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)



def clean_text(list):

    stop = stopwords.words('english')
    list = list.apply(lambda x : replace_contractions(x))
    list = list.str.replace("[^a-zA-Z0-9]+",' ')
    list = list.apply(lambda x: " ".join(x.lower() for x in x.split()))
    list = list.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    #lemmatization
    list = list.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return list

def tfidf_vectorizer(df, column):
  tfv = TfidfVectorizer(dtype=np.float32, min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1,5), use_idf=1,smooth_idf=1,sublinear_tf=1,
           # stop_words = 'english'
                     )
 
  tfidf_vect_fit=tfv.fit(df[column])
  tfidf = tfv.transform(df[column])
  words = tfv.get_feature_names()
  tfidf_df = pd.DataFrame(tfidf.toarray())
  tfidf_df.columns = words
  return tfidf_df, tfv

  
def TrunSVD(matrice):
  svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)
  svd_df = svd.fit_transform(matrice)
  return svd_df, svd

def Word2vec(data):
  texts =data.values
  sentences = [nltk.word_tokenize(text) for text in texts]
  model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4)
  return model

def convert_to_vec(model,X_train ,X_test):
  words = set(model.wv.index_to_key )
  X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words])
                         for ls in X_train])
  X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words])
                         for ls in X_test])
  X_train_vect_avg = []
  for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
        
  X_test_vect_avg = []
  for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))
  return X_train_vect_avg,X_test_vect_avg

def encoding_target(column):
    column =column.map({'anger':0, 'disgust':1,'fear':2,'joy':3, "neutral":4,"sadness":5})
    return column


def split_data(corpus, target, test_size):
  X_train, X_test, y_train, y_test = train_test_split(corpus,target,test_size = test_size, random_state=42, stratify=target)
  return X_train, X_test, y_train, y_test
