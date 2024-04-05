'''
Author: Chen Wei
'''
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import seaborn as sns

SVB_MARCH_PATH = "../data/after_clean/SVB Clean.csv"
SUSSIE_MARCH_PATH = "../data/after_clean/Credit Sussie Clean.csv"
WORLD_MARCH_PATH = "../data/world_march_labeled_cleaned.csv"


def bar_date_distribution(table_name, path):

    # fig, ax = plt.subplots()
    df = pd.read_csv(path)
    plt.figure(figsize=(12, 6))

    # transform date to string
    df['Date'] = df['Date'].apply(lambda x: str(x).split(' ')[0])
    
    # select date with correct format
    pattern = re.compile(r'^\d{4}-\d{2}-\d{2}')
    df = df[df['Date'].apply(lambda x: pattern.match(x) != None)]

    vc = df['Date'].value_counts().sort_index()
    x_coords = range(len(vc.index))
    plt.plot(x_coords, vc.values, color='lightgreen', linestyle='-', linewidth=2, marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, markeredgecolor='orange')
    plt.xticks(x_coords, vc.index, rotation=30, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Date", fontsize=14, labelpad=10)
    plt.ylabel("Frequency", fontsize=14, labelpad=10)
    plt.title('Twitter Date Frequency Plot ' + table_name, fontsize=16, pad=20)
    plt.grid(True, axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
    plt.savefig('../plot/' + table_name + ' Twitter Date Distribution Plot')
    plt.show()

def bar_sentiment_distribution(table_name, path):
    df0 = pd.read_csv(path[0], index_col=0)
    df1 = pd.read_csv(path[1], index_col=0)
    df2 = pd.read_csv(path[2], index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.set_size_inches(18, 5)
    axs[0].hist(df0['label'], bins=np.arange(4)-0.5)
    axs[0].set_title(path[0].split('/')[-1])
    axs[1].hist(df1['label'], bins=np.arange(4)-0.5)
    axs[1].set_title(path[1].split('/')[-1])
    axs[2].hist(df2['Overall_Sentiment'].replace({0: "negative", 1: "positive", 2: "neutral"}), bins=np.arange(4)-0.5)
    axs[2].set_title(path[2].split('/')[-1])
    plt.savefig('../plot/' + table_name + ' Twitter Sentiment Distribution Plot (separate)')
    plt.show()

def bar_sentiment_distribution_sns(table_name, path):
    df0 = pd.read_csv(path[0], index_col=0)
    vc0 = df0['label'].rename("label")
    type0 = path[0].split('/')[-1].split('.')[0]
    df1 = pd.read_csv(path[1], index_col=0)
    vc1 = df1['label'].rename("label")
    type1 = path[1].split('/')[-1].split('.')[0]
    df2 = pd.read_csv(path[2], index_col=0)
    vc2 = df2['Overall_Sentiment'].replace({0: "negative", 1: "positive", 2: "neutral"}).rename("label")
    type2 = path[2].split('/')[-1].split('.')[0] 
    df = pd.concat((
        pd.concat((vc0, vc1, vc2)).reset_index(drop=True),
        pd.concat((pd.Series([type0] * len(vc0), name='csv'), pd.Series([type1] * len(vc1), name='csv'), pd.Series([type2] * len(vc2), name='csv'))).reset_index(drop=True)
    ), axis=1)
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="label", hue="csv", element="step")

    plt.savefig('../plot/' + table_name + ' Twitter Sentiment Distribution Plot (together)')
    plt.show()


##############################################################################
'''Main'''
##############################################################################
bar_date_distribution('World', WORLD_MARCH_PATH)
bar_sentiment_distribution('World_SVB_Sussie', [SVB_MARCH_PATH, SUSSIE_MARCH_PATH, WORLD_MARCH_PATH])
bar_sentiment_distribution_sns('World_SVB_Sussie', [SVB_MARCH_PATH, SUSSIE_MARCH_PATH, WORLD_MARCH_PATH])