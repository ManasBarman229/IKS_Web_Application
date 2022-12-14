
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import string as st
import numpy as np
import pandas as pd
# import snscrape.modules.twitter as sntwitter
# import itertools


df = pd.read_csv('dataset.csv', encoding="utf8")


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
nltk.download('stopwords')
stoplist = stopwords.words('english')


def remove_stopwords(text):
    return [word for word in text if word not in stoplist]


df['clean_tokens'] = df['filtered_tokens'].apply(lambda x: remove_stopwords(x))


# Create sentences to get clean text as input for vectors


def return_sentences(tokens):
    return " ".join([word for word in tokens])


df['clean_text'] = df['clean_tokens'].apply(lambda x: return_sentences(x))


# Prepare data for the model. Convert label in to binary

df['label'] = [0 if x == 'No' else 1 for x in df['label']]


# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.1)


# vectorization

tfidf = TfidfVectorizer()
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)


# Naive Bayes model

nb = GaussianNB()
nb.fit(tfidf_train.toarray(), y_train)


preb = nb.predict(tfidf_train.toarray())


pred = nb.predict(tfidf_test.toarray())


# function for manual data query
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
    if (result == 1):
        display_value = "yes,It is Indigenous"
    else:
        display_value = "no, It is not Indigenous"
    return display_value


# twitter scrapper

# text_query = hash_input+'' + \
#     'lang:en  geocode:21.1458,79.0882,2000km exclude:links exclude:mentions exclude:hashtags'
# df = pd.DataFrame(itertools.islice(
#     sntwitter.TwitterSearchScraper(text_query).get_items(), 100))


# output = pd.DataFrame
# output = df.loc[:, ("rawContent", "url", "date")]
# output.rename(columns={'rawContent': 'text'}, inplace=True)

# output.to_csv(r'files\\assamese.csv', index=False)


def getData(hash_input):
    # text_query = hash_input+" "+"-filter:retweets"
    # count = 100
    # try:
    #     tweets_obj = tweepy.Cursor(
    #         api.search_tweets, q=text_query).items(count)
    #     tweets_list = [[tweet.text] for tweet in tweets_obj]
    #     f = pd.DataFrame(tweets_list)

    # except BaseException as e:

    #     print("something went wrong, ", str(e))
    f = pd.read_csv('cultural.csv', encoding="utf8")
    f.rename(columns={0: 'text'}, inplace=True)
    f = f.fillna(' ')
    f['removed_punc'] = f['text'].apply(lambda x: remove_punct(x))
    f['tokens'] = f['removed_punc'].apply(lambda x: tokenize(x))
    f['filtered_tokens'] = f['tokens'].apply(lambda x: remove_small_words(x))
    f['clean_tokens'] = f['filtered_tokens'].apply(
        lambda x: remove_stopwords(x))
    f['clean_text'] = f['clean_tokens'].apply(lambda x: return_sentences(x))
    tfidf_f = tfidf.transform(f['clean_text'])
    mo = nb.predict(tfidf_f.toarray())  # 'mo' is the array of outputs(0&1)
    f.drop(['removed_punc', 'tokens', 'filtered_tokens',
            'clean_tokens', 'clean_text'], axis='columns', inplace=True)
    f['Output'] = mo
    f.replace(1, 'Yes', inplace=True)
    f.replace(0, 'No', inplace=True)
    data = f.values.tolist()
    # data.to_csv(
    #     r'C:\\Users\\Manas\\Desktop\\Project\\CSV\\ayurdeva.csv', index=False)
    return data


# getData("Ayurveda")
