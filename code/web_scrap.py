'''
Author: Chen Wei
'''
import sqlite3
import pandas as pd
# # ps: could also use twitterscraper to scrapt the twitter, but has limitation of 3000 twitters and only 7 days
import snscrape.modules.twitter as sntwitter  # used for scrapping data, required python 3.8+ env

NUM_TWEET = 10000  # tweet number limit
WORD_ECONOMY_TWEET = 100000

# SVB 3.10 collapse
SVB_MARCH_QUERY = "SVB lang:en until:2023-03-11 since:2023-03-9"   # language: english
# Credit Sussie 3.19 being took over
SUSSIE_MARCH_QUERY = "Credit Suisse lang:en until:2023-03-20 since:2023-03-18"   # found that 1 day ≈ 10000 data, so maybe cannot get that much
# World economy query
WORLD_ECONOMY_QUERY = "world economy lang:en until:2023-03-20 since:2023-03-09"

DB_PATH = "../data/tweets_data.db"


def scrapping(query):
    tweets = []

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():

        if len(tweets) == NUM_TWEET:
            break
        else:
            # Todo: double check the api
            tweets.append([tweet.id, tweet.user.username, tweet.content, tweet.likeCount, tweet.quoteCount, tweet.replyCount, tweet.url, tweet.date ])

    tweet_df = pd.DataFrame(tweets, columns = ['Tweet_id', 'User', 'Content', "Like_Count", "Quote_Count", "Reply_Count", "Url", 'Date'])
    tweet_df.dropna()

    # print(tweet_df.head())
    return tweet_df


# Store it in the data.db
def dataframe_to_db(df, target_form):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Delete tables if they exist
    c.execute ('DROP TABLE IF EXISTS "tweets";')
    # create stock table

    # Todo: join the data (maybe? and group by date for tweets) - TBD for further analysis, this is why leave the tweets
    c.execute(
        'CREATE TABLE IF NOT EXISTS tweets (Id text not null, User text, Content text, Like int, Quote int, Reply int, Url text, Date date, PRIMARY KEY (Id))'
    )

    df.to_sql(target_form, conn, if_exists='replace', index = False)
    conn.commit()

# Store it in the csv
def dataframe_to_csv(df, target_csv):
    df.to_csv('../data/' + target_csv + '.csv')



#######################################################################################################################
'''Main scrape pipeline'''
#######################################################################################################################
# Step 1: get svb 2023.3 (actually 3.9 - 3.10 due to the limit) tweet data
svb_march_df = scrapping(SVB_MARCH_QUERY)
dataframe_to_db(svb_march_df, "svb_march")
dataframe_to_csv(svb_march_df, "svb_march")

# Step 2: get credit sussie 2023.3 (actually 3.18 - 3.19 due to the limit) tweet data
sussie_march_df = scrapping(SUSSIE_MARCH_QUERY)
dataframe_to_db(sussie_march_df, "sussie_march")
dataframe_to_csv(sussie_march_df, "sussie_march")

# Step 3: get world economy 2023.3 (actually 3.1 - 3.31)
world_march_df = scrapping(WORLD_ECONOMY_QUERY)
dataframe_to_db(world_march_df, "world_march_not_labeled")
dataframe_to_csv(world_march_df, "world_march_not_labeled")