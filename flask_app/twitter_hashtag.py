import pandas as pd
import tweepy as tw
# Twitter API key and API secret
my_api_key = "Zq3KVYsMbUibAKNWFwecP62YE"
my_api_secret = "PR1BAPVZjlrKny8V93oKxNAXOiGdlSMpyTbQt7BP0PRClwGRPx"

# authenticate
auth = tw.OAuthHandler(my_api_key, my_api_secret)
api = tw.API(auth, wait_on_rate_limit=True)

search_query = "#ayurveda -filter:retweets"

# tweets from the API
tweets = tw.Cursor(api.search_tweets,
                   q=search_query,
                   lang="en",
                   since="2021-09-16").items(100)

# store the API responses in a list
tweets_copy = []
for tweet in tweets:
    tweets_copy.append(tweet)

print("Total Tweets fetched:", len(tweets_copy))


# intialize the dataframe
tweets_df = pd.DataFrame()

# populate the dataframe
for tweet in tweets_copy:
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    tweets_df = tweets_df.append(pd.DataFrame({
        'date': tweet.created_at,
        'text': text,
        'hashtags': [hashtags if hashtags else None],
    }))
    tweets_df = tweets_df.reset_index(drop=True)

# show the dataframe
tweets_df.head()

tweets_df.drop(["date", "hashtags"], axis=1, inplace=True)
print(tweets_df)
tweets_df.to_csv(
    r'C:\\Users\\Manas\\Desktop\\Project\\CSV\\twitter_hashtag.csv', index=False)
