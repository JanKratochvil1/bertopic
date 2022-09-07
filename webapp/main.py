from fastapi import FastAPI, Response
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model, language='english', diversity = 0.4, top_n_words=10, n_gram_range=(1, 3), min_topic_size=6)

def split_sentences(text):
    sentences = text.split('.')
    return sentences

def clean_initial(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text.lower() # Lower Case
    tokenized = word_tokenize(lowercased) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    return lemmatized

def sentences_to_clean_sentences_initial(sentences):
    sentences_clean = []
    counter = 0
    for item in sentences:
        sentences_clean.append(clean_initial(sentences[counter]))
        counter = counter + 1

    counter = 0
    for item in sentences_clean:
        sentences_clean[counter] = ' '.join(sentences_clean[counter])
        counter = counter + 1
    return sentences_clean


app = FastAPI()

class Body(BaseModel):
    text: str

@app.get('/')
def root():
    return Response("<h1>An API to interact with custom Bertopic</h1>")


@app.post('/get_topics')
def get_topics(body: Body):
    sentences = split_sentences(body.text)
    sentences_clean = sentences_to_clean_sentences_initial(sentences)
    topics, probs = topic_model.fit_transform(sentences_clean)
    counter = 0
    list = []
    for i in topic_model.get_topic_info()['Name'][1:11]:
        list.append(i)
    print(list)
    return list
