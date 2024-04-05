'''
Author: Chen Wei
'''
import json
import numpy as np
import pandas as pd
import re
import string
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')

sns.set(style = "darkgrid")

SVB_MARCH_PATH = "../data/svb_march.csv"
SUSSIE_MARCH_PATH = "../data/sussie_march.csv"
WORLD_MARCH_PATH = "../data/world_march.csv"

PRONONS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
           "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
           "theirs", "themselves"]
PROP = ["of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "the"]
# Todo: Take a closer look at the take and decide to remove more stopwords
AMOUNT = ["a", "an", "many", "much"]

OTHERS = ["svb", "world", "sussie"]


def reg_clean_helper(txt):
    txt = re.sub('\n', '', txt)
    txt = re.sub('\[.*?\]', '', txt)
    txt = re.sub('\w*\d\w*', '', txt)
    txt = re.sub('<.*?>+', '', txt)
    txt = re.sub('https?://\S+|www\.\S+', '', txt)
    txt = re.sub('\w*\d\w*', '', txt)
    return txt

# Some data cleaning, selection
def processing(path):
    # Read data from csv file
    march_df = pd.read_csv(path)
    print("\n\n  march data raw samples:", march_df.head(10))

    # Tweet_id,User,Content,Like_Count,Quote_Count,Reply_Count,Url,Date
    selected_cols = ['User', 'Content', 'Like_Count', 'Quote_Count', 'Reply_Count', 'Date']
    selected_df = march_df[selected_cols]

    # remove the specific time, only leave the date
    selected_df.Date = pd.to_datetime(selected_df.Date).dt.date
    # Assign a unique numerical code to each category
    selected_df.User = selected_df.User.astype('category')
    selected_df.User = selected_df.User.cat.codes

    # removing URLs from tweets, make it all lower case and regularize it
    selected_df['content_words'] = selected_df['Content'].apply(lambda x: re.sub(r'http\S+', '', str(x)).lower())
    selected_df['content_words'] = selected_df['content_words'].apply(lambda x: reg_clean_helper(x))

    # remove punctuations and stopwords and add it to a new column
    # Todo: check the stop word
    stop_word = set(stopwords.words('english'))
    stop_word.update(PRONONS)
    stop_word.update(PROP)
    stop_word.update(AMOUNT)
    stop_word.update(OTHERS)

    punc_filter = lambda x:x.translate(str.maketrans('','',string.punctuation))
    stopwords_filter = lambda x: ' '.join([word for word in x.split() if word not in stop_word])

    selected_df['content_words'] = selected_df['content_words'].apply(punc_filter)
    selected_df['content_words'] = selected_df['content_words'].apply(stopwords_filter)
    selected_df['words'] = selected_df['content_words'].apply(lambda x: x.split( ))
    print("\n\n  march data after cleaning samples:", march_df.head(10))

    all_words = [word for lines in selected_df['content_words'] for word in lines.split()]

    return selected_df, all_words

# Common words bar plot
def bar_plot_common_words(table_name, all_words):
    word_counter = Counter(all_words).most_common(50)
    helper_df = pd.DataFrame(word_counter)
    helper_df.columns = ['word', 'freq']

    fig = plt.figure(figsize = (18, 7))

    plt.xticks(rotation = 90)
    plt.bar(helper_df['word'], helper_df['freq'])
    plt.xlabel('word')
    plt.ylabel('frequency')
    plt.title('50 Most Common Words in ' + table_name)
    # plt.show()
    plt.savefig('../plot/' + table_name + ' 50 common words')


def cal_sentiment_scores(df):
    si_analyzer = SentimentIntensityAnalyzer()
    senti_scores = df['content_words'].apply(lambda x: si_analyzer.polarity_scores(x))
    senti_df = pd.DataFrame(list(senti_scores))
    senti_df['label'] = senti_df['compound'].apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))
    print(senti_df.head(10))
    return senti_df


# Todo: if doing sentiment analysis, add some columns (We would probably using nltk)  [is it really needed, as we have nltk and hypothesis to help us do this]
def count_pos_negative_word(content):
    pass


# Todo: add more
def pipeline(path, name):
    cleaned_df, march_worlds = processing(path)
    print("\n\n" + name + " cleaned dataframe:\n ", cleaned_df)
    print("\n\n" + name + " march top 50 twitter words: \n", march_worlds[:50])
    bar_plot_common_words(name + " March Twitter", march_worlds)
    senti_df = cal_sentiment_scores(cleaned_df)
    senti_data = cleaned_df.join(senti_df)
    # senti_counts = senti_data['label'].value_counts()
    print(senti_data.head(10))
    senti_data.to_csv('../data/after_clean/' + name + ' Clean.csv')





#######################################################################################################################
'''Main'''
#######################################################################################################################
pipeline(SVB_MARCH_PATH, "SVB")
pipeline(SUSSIE_MARCH_PATH,"Credit Sussie")
pipeline(WORLD_MARCH_PATH, "World Economy")


# Todo(?): maybe a way to read from database as well if we find join useful



# Todo: count negative, neural, positive (can be counted as analysis)