from heapq import heapify
import os
import json
from typing import Counter
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
import sklearn.model_selection
import sklearn.preprocessing as preproc
from sklearn.feature_extraction import text
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline

from tqdm import tqdm

# from rouge_score import rouge_scorer

def json_to_df(json_path,type):
  with open(json_path, "r", encoding="utf-8") as f: 
    lines = [eval(l) for l in f.readlines()]

  # exclude lines with surrogates in their text/summary
  surr = [ i for i,l in enumerate(lines) for k in l.keys() if k in ["text","summary"] and re.search(r'[\uD800-\uDFFF]', l[k])!=None ]
          
  lines = [ l for i,l in zip( range(len(lines)),lines ) if i not in surr ]

  cols=[ "title",	"date",	"text",	"summary", "compression", "coverage", "density", "compression_bin", "coverage_bin"]

  # we need only the extractive summaries as we are building an extractive summarizer
  data=[ [ l[k] for k in l.keys() if k in cols ] for l in lines if l["density_bin"]=="extractive" ]
  df = pd.DataFrame(data,columns=cols)

  df.to_csv(f"..{os.sep}Data{os.sep}DataFrames{os.sep}{type}_set.csv", header=True, index=False )

  return df


# text processing functions

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "must've": "must have", "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "that'd": "that would", "that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are" }


def sentence_cleaning(text, remove_stopwords = True):
  # Convert words to lower case
  text = text.lower()

  # Replace contractions with their longer forms 
  if True:
    text = word_tokenize(text)
    new_text = []
    for word in text:
      if word in contractions:
        new_text.append(contractions[word])
      else:
        new_text.append(word)
    
    text = " ".join(new_text)

  # Format words and remove unwanted characters
  text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  text = re.sub(r'\<a href', ' ', text)
  text = re.sub(r'&amp;', '', text) 
  text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
  text = re.sub(r'<br />', ' ', text)
  text = re.sub(r'\'', ' ', text)

  # remove stop words
  if remove_stopwords:
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)

  # Tokenize each word
  # text =  nltk.WordPunctTokenizer().tokenize(text)
      
  return text


def rouge_scoring(sentence,summary,type="rougeL",score="fmeasure"):
  global pbar
  pbar.update(1)
  r_scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"])
  r_L=r_scorer.score(summary,sentence)
  score_ind={"precision":0, "recall":1, "fmeasure":2}

  return r_L[type][score_ind[score]]


def text_processing_old(df,type):
  global pbar
  cols=["sentence", "summary", "text"] 
  # new_df=pd.DataFrame()

  # sentence split 
  sentences=[ [ sentence_cleaning(s) for s in sent_tokenize(t)] for t in df["text"].values ]
  summaries=[ [ sentence_cleaning(s) for s in sent_tokenize(t)] for t in df["summary"].values ]
  summaries=[ ".".join(s) for s in summaries ]
  
  sent_sum_text=[ [ s,summary,t  ] for s_list,summary,t in zip( sentences, summaries, df["text"] ) for s in s_list ]
  new_df=pd.DataFrame(sent_sum_text, columns=cols)
  # for c in new_df.columns:
  #   new_df[c]=new_df[c].astype(str)

  # sentence cleaning & tokenization
  # sentences_clean = [  [ sentence_cleaning(s) for s in s_list ]  for s_list in sentences ]

  # sentence feature representation

  # labels
  # columns -> sentence: 0, summary: 1, text: 2
  pbar = tqdm(int(new_df.shape[0]) )
  new_df["rougeL"]= new_df.apply(lambda row: rouge_scoring(row["sentence"],row["summary"], type="rougeL", score="fmeasure" ), axis=1)
  print(new_df["rougeL"])

  new_df.to_csv(f"..{os.sep}Data{os.sep}DataFrames{os.sep}{type}_data.csv", header=True, index=False)

  return new_df




# Classifiers
SVM_scaler =  StandardScaler()
LR_scaler =  MinMaxScaler()
KNN_scaler =  StandardScaler()
# classifier parameters
KNN_n_num = 9
LR_C = 1.0
SVM_C = 0.001
algos={
  "SVM": make_pipeline(SVM_scaler, SVC(C=SVM_C)),
  "LR":  make_pipeline(LR_scaler, LogisticRegression(C=LR_C)),
  "KNN": make_pipeline(KNN_scaler, KNeighborsClassifier(n_neighbors = KNN_n_num)),
}


def classifier_training(model,X_train,y_train):
  model.fit(X_train,y_train)
  preds=model.predict(X_train)
  c_rep=classification_report(y_train,preds)
  c_rep_dict=classification_report(y_train,preds,output_dict=True)
  return model, c_rep, c_rep_dict

def classifier_validation(model,X_dev,y_dev):
  preds=model.predict(X_dev)
  c_rep = classification_report(y_dev,preds)
  c_rep_dict=classification_report(y_dev,preds,output_dict=True)
  return c_rep, c_rep_dict

def classifier_test(model,X_test,y_test):
  preds=model.predict(X_test)
  c_rep = classification_report(y_test,preds)
  c_rep_dict=classification_report(y_test,preds,output_dict=True)
  return c_rep, c_rep_dict


# Training - Validation - Test pipeline
def classifier_T_V_T(X_train, y_train, X_dev, y_dev, X_test, y_test, algo_type="LR"):
  model=algos[algo_type]

  model,c_rep_train,c_rep_dict_train=classifier_training(model,X_train,y_train)
  c_rep_dev,c_rep_dict_dev=classifier_validation(model,X_dev,y_dev)
  c_rep_test,c_rep_dict_test=classifier_validation(model,X_test,y_test)

  return model, c_rep_train, c_rep_dict_train, c_rep_dev, c_rep_dict_dev, c_rep_test, c_rep_dict_test


def text_processing(df,data_type,df_dir,sc_type="rougeL"):
  global pbar
  cols=["sentence", "summary", "text", "text_id"] 

  df["summary"].to_csv(os.path.join(df_dir,f"{data_type}_data_{sc_type}_summaries_grouped.csv"), header=True, index=False)
  #     print(f"Summary df len: {df['summary'].shape}")
  # new_df=pd.DataFrame()

  # sentence split 
  sentences=[ sent_tokenize(t) for t in df["text"].values ]
  #     print(f"Sentence Len: {len(sentences)}")

  summaries=df["summary"].values
  sent_sum_text=[ [ s,summary,t,i  ] for i, (s_list,summary,t) in enumerate(zip( sentences, summaries, df["text"] )) for s in s_list ]
  new_df=pd.DataFrame(sent_sum_text, columns=cols)
  #     new_df["text_id"]=new_df["text"].factorize()[0]

  #     print(len(sent_sum_text))
  #     print(set(new_df["text_id"].values), set(new_df["t_id"].values))
  #     print(new_df["text_id"].values[-1])

  new_df["chosen"]= 0
  ind = new_df[[ s in t for s,t in zip( new_df["sentence"], new_df["summary"] ) ]].index
  new_df.loc[ind,"chosen"]=1
  del new_df["text"]
  # for c in new_df.columns:
  #   new_df[c]=new_df[c].astype(str)

  # labels
  # columns -> sentence: 0, summary: 1, text: 2
  pbar = tqdm(total=new_df.shape[0] )
  new_df["rougeL"]= new_df.apply(lambda row: rouge_scoring(row["sentence"],row["summary"], sc_type=sc_type, score="fmeasure" ), axis=1)
  #     print(new_df["rougeL"])

  new_df["summary"].to_csv(os.path.join(df_dir,f"{data_type}_data_{sc_type}_summaries.csv"), header=True, index=False)
  del new_df["summary"]

  new_df.to_csv(os.path.join(df_dir,f"{data_type}_data_{sc_type}.csv"), header=True, index=False)

  return new_df


def json_to_df(json_path,data_type):
  data=[]
  for ln in open(json_path,"r"):
      obj = json.loads(ln)
      data.append(obj)
  df=pd.DataFrame(data)

  cols=[ "title",	"date",	"text",	"summary", "compression", "coverage", "density", "compression_bin", "coverage_bin"]
  df=df.loc[df.density_bin=="extractive"].reset_index()

  df.to_csv(f"..{os.sep}..{os.sep}Data{os.sep}DataFrames{os.sep}{data_type}_set.csv", header=True, index=False )
  df["summary"].to_csv(f"..{os.sep}..{os.sep}Data{os.sep}DataFrames{os.sep}{data_type}_summaries.csv", header=True, index=False )

  return df


if __name__=="__main__":
  data_dir=f"..{os.sep}..{os.sep}Data{os.sep}DataFrames"
  data_type="train"
  train_df = json_to_df(f"..{os.sep}..{os.sep}Data{os.sep}release{os.sep}{data_type}.jsonl",data_type)
  splits=4
  train_data_list=[]
  for i,train_df in enumerate(np.array_split(train_df, splits)):
    train_df.to_csv(f"..{os.sep}..{os.sep}Data{os.sep}DataFrames{os.sep}{data_type}{i+1}_set.csv", header=True, index=False )
    train_data_list.append(train_df)

