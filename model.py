
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
import re
import string as st
import numpy as np
import pandas as pd


df = pd.read_csv(
    'C:\\Users\\Manas\\Desktop\\Project\\CSV\\proto-2.csv', encoding='windows-1252')


df.isnull().sum()

df = df.fillna(' ')


# Remove all punctuations from the text


def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))


df['removed_punc'] = df['text'].apply(lambda x: remove_punct(x))

# Convert text to lower case tokens.
# Split() is applied on white-spaces to seperate into tokens.


def tokenize(text):
    text = re.split('\s+', text)
    return [x.lower() for x in text]


df['tokens'] = df['removed_punc'].apply(lambda msg: tokenize(msg))


# Remove tokens of length less than 3


def remove_small_words(text):
    return [x for x in text if len(x) > 2]


df['filtered_tokens'] = df['tokens'].apply(lambda x: remove_small_words(x))


# Remove stopwords. Here, NLTK corpus list is used for a match.
# nltk.download('stopwords')


def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]


df['clean_tokens'] = df['tokens'].apply(lambda x: remove_stopwords(x))


# Create sentences to get clean text as input for vectors


def return_sentences(tokens):
    return " ".join([word for word in tokens])


df['clean_text'] = df['clean_tokens'].apply(lambda x: return_sentences(x))


# Prepare data for the model. Convert label in to binary

df['label'] = [1 if x == 'No' else 0 for x in df['label']]


# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.1)


"""***TF-IDF : Term Frequency - Inverse Document Frequency***"""

# vectorization

tfidf = TfidfVectorizer()
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

# """***Classifiers***"""


# Naive Bayes model

nb = GaussianNB()
nb.fit(tfidf_train.toarray(), y_train)


preb = nb.predict(tfidf_train.toarray())


pred = nb.predict(tfidf_test.toarray())


#d='turmeric is good for health'
#d=d.fillna(' ')


def manual_query_input(indi_data):
    d = indi_data
    d = remove_punct(d)
    d = tokenize(d)
    d = remove_small_words(d)
    d = remove_stopwords(d)
    d = return_sentences(d)
    d = [d]
    tfidf_d = tfidf.transform(d)

    result = nb.predict(tfidf_d.toarray())  # 0 means indigenous, 1 means not
    if (result == 0):
        display_value = "yes,It is Indigenous"
    else:
        display_value = "no, It is not Indigenous"
    return display_value
