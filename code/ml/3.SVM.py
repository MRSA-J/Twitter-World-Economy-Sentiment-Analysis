'''
@ Author MinFeiXue Zong
'''

import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


###read file
def dataread(file):
    we_df = pd.read_csv(file)
    data = we_df.dropna()
    data['Content'] = data['Content'].str.lower()
    return data

data_file = "../../data/world_march_labeled_cleaned.csv"

data = dataread(data_file)

dataread(data_file)

###use nltk to split the comment
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
additional_words = set(["...", "’", "“", "”"])

def remove_stopwords_punctuation(text):
    ### Remove non-alphanumeric characters and replace them with spaces
    text = re.sub(r'\W', ' ', str(text))

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    words = word_tokenize(text)
    ### Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    stop_words = set(stopwords.words("english"))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

data['Cleaned_Content'] = data['Content'].apply(remove_stopwords_punctuation)

sia = SentimentIntensityAnalyzer()

def sentiment_label(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

data['Sentiment'] = data['Cleaned_Content'].apply(lambda x: sia.polarity_scores(x)['compound'])
data['Sentiment_Label'] = data['Sentiment'].apply(sentiment_label)

X = data['Cleaned_Content']
y = data['Sentiment_Label']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

###print the report
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
