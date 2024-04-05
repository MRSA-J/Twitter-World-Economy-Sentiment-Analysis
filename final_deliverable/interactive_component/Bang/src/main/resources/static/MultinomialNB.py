"""
@ Author: ZhenHao Sun
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.corpus import stopwords
# nltk.download('wordnet')   #if there is an error loading it, uncomment this to download the nltk wordnet
from nltk.stem import WordNetLemmatizer

import re
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle
import os

BEST_MODEL_PATH = "/Users/sunzhenhao/IdeaProjects/Bang/src/main/resources/static/best_model.pkl"
VECTORIZER_PATH = "/Users/sunzhenhao/IdeaProjects/Bang/src/main/resources/static/vectorizer.pkl"
WORLD_ECONOMY_DATA_PATH = "./data/world_march_labeled_cleaned.csv"

def load_file(file_path):
    ### load files
    data = pd.read_csv(file_path)
    data.dropna()
    return data

def preprocess_text(text):
    ### use online stop words as split signal
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Training pipeline
def get_best_model(is_saved = True):
    ### load data
    data = load_file(WORLD_ECONOMY_DATA_PATH)

    ### preprocess the context column
    data['processed_Content'] = data['Content'].apply(preprocess_text)

    ### create tf-idf dataset
    X = data['processed_Content']
    y = data['Overall_Sentiment']
    vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
    X_tfidf = vectorizer.fit_transform(X).toarray()

    ### we balance the dataset using the Synthetic Minority Over-sampling Technique (SMOTE) to address the class imbalance issue in the dataset
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_tfidf, y)

    ### split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    print("X_train")

    ### use gridsearch method to get the most accurate model
    pipeline = Pipeline([
        ('classifier', MultinomialNB())
    ])

    param_grid_multinomialNB = {
        'classifier': [MultinomialNB()],
        'classifier__alpha': np.linspace(0.1, 2.0, 20)
    }
    param_grid_logisticRegression = {
        'classifier': [LogisticRegression(solver='liblinear')],
        'classifier__C': np.logspace(-3, 3, 7),  # Regularization parameter
        'classifier__penalty': ['l1', 'l2']  # Regularization type
    }
    # Combine the parameter grids into a list
    parameters = [param_grid_multinomialNB, param_grid_logisticRegression]

    ### perform the gird search
    grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    ### Print the best parameters
    print("Best parameters found:")
    print(grid_search.best_params_)

    y_test_pred = grid_search.predict(X_test)
    y_train_pred = grid_search.predict(X_train)

    if is_saved:
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(grid_search.best_estimator_, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)

    ### Print related attributes
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
    print("\nAccuracy Score for testing:", accuracy_score(y_test, y_test_pred))
    print("\nAccuracy Score for trainning:", accuracy_score(y_train, y_train_pred))


def predict(words):
    current_path = os.getcwd()
    # Load the best model from the file
    with open(BEST_MODEL_PATH, "rb") as f:
        grid_search = pickle.load(f)

    # Load the saved vectorizer
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    # Preprocess the new tweet
    processed_tweet = preprocess_text(words)

    # Convert the processed tweet to a TF-IDF feature matrix
    tweet_tfidf = vectorizer.transform([processed_tweet]).toarray()

    # Predict the sentiment of the new tweet
    sentiment_prediction = grid_search.predict(tweet_tfidf)

    # Map the numeric labels back to their original text labels
    label_map = {0: "negative", 1: "positive", 2: "neutral"}

    # Print the predicted sentiment
    res = sentiment_prediction[0]
    return res

##############################################################################################################
'''Main'''
##############################################################################################################
if __name__ == '__main__':
    arg = input()
    res = predict(arg)
    print(res)
