
import tweepy
import pandas as pd

consumer_key = "Zq3KVYsMbUibAKNWFwecP62YE"
consumer_secret = "PR1BAPVZjlrKny8V93oKxNAXOiGdlSMpyTbQt7BP0PRClwGRPx"

access_token = "1554767526067728385-wz0xP9nMUqV5I9roUIP98ZGN1LpAOF"

access_token_secret = "4W6b8e0dNQElXJYxf2RKahSFRwv76k90ir99MPuYpNllf"

authorization = tweepy.OAuthHandler(consumer_key, consumer_secret)

authorization.set_access_token(access_token, access_token_secret)

api = tweepy.API(authorization, wait_on_rate_limit=True)


# def setKeyword(input_key):
#     temp = input_key+" "+"-filter:retweets"
#     return temp


# setKeyword("hello")
# text_query = setKeyword(input_key)
# print(text_query)

text_query = "arigato  -filter:retweets"

count = 100
df = pd.DataFrame()
try:
    tweets_obj = tweepy.Cursor(api.search_tweets, q=text_query).items(count)
    tweets_list = [[tweet.text] for tweet in tweets_obj]
    df = pd.DataFrame(tweets_list)

except BaseException as e:

    print("something went wrong, ", str(e))
df.rename(columns={0: 'text'}, inplace=True)


# print(df)


# def getScrapData():
#     return df


df.to_csv(
    r'C:\\Users\\Manas\\Desktop\\Project\\CSV\\arigato.csv', index=False)
