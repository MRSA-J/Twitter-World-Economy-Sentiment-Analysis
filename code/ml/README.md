# ML


----
### 1. LDA
LDA is a type of unsupervised learning that can analyze and summarize a text corpus without relying on labeled data, making it ideal for exploration and summarization even if there is no prior knowledge of the corpus content. We used this model to find high frequency topics in Twitter.

**How?** <br>
The preprocessing steps involve lowercasing, tokenization, stopword removal, and lemmatization. This ensures consistency, splits the text into individual words, removes common words that don't add much meaning, and converts words to their base forms to reduce the feature space and enhance model performance. The LDA model is trained on the resulting corpus, and the discovered topics are printed. The trained model is also saved for later use, and the pyLDAvis library is used to visualize the topics.

**Parameter Setting** <br>
num_topics: The number of topics to be discovered. This is set to 5 in the code, but it can be changed according to the requirements.<br>
id2word: The dictionary of the corpus. This is set to the dictionary object created from the preprocessed tokens.<br>
passes: The number of passes over the corpus during training. This is set to 10 in the code.<br>
random_state: The random seed used for the LDA model. This is set to 42 in the code.<br>

**Run the model** <br>
First, cd to the `ml` folder. <br>

Then, run:
```
python 1.LDA.py
```

Before running the code, we will need to make sure that the required libraries (pandas, re, spacy, gensim, chardet, and pyLDAvis) are installed. We can install these libraries using pip. Once we have made the necessary changes, we can run the code to preprocess the text data, train the LDA model, and print the discovered topics. The trained model will be saved as a file named lda_model.model in the current working directory.

**Evaluation Metric** <br>

The model is evaluated from two important perspectives: topic coherence and interpretability. Topic coherence measures the semantic similarity between the top terms within a topic, and higher coherence indicates more interpretable topics. Interpretability refers to the human interpretation and judgment of the generated topics' quality and the ability to understand the underlying themes in the data. 

**Sample prediction** <br>
The results of the topic modeling show that the top two common topics are related to the war's impact on the world economy and international relationships. The third topic is more focused on monetary policies and the world economy. The fourth and fifth topics are less established, but they give insights into people's attitudes towards the world economy, such as the need for "change", "control", and opinions about "failure". Overall, the model's predictions are satisfactory. <br>

###### A visualization can be found [here](https://chenxiwub.github.io/ldaplot/#topic=0&lambda=1&term=).  

### 2. Multinomial Naive Bayes Model
We used Multinomial Naive Bayes Model because this model is particularly suited text classification tasks as it handles discrete data well.

**How?** <br>
The data is transformed into a bag-of-words representation using the TF-IDF vectorizer, which results in a count-based feature representation. Multinomial Naive Bayes can model the frequency of words in each class effectively, making it a suitable choice for sentiment analysis 

**Parameter Setting** <br>
There are two parts I need to adjust my parameters. <br>
The first is about the `TfidfVectorizer`. My way of deciding the specific attributes is based on my analysis on the dataset. I will adjust the parameters according to my understanding of the dataset. After I obtain the rough attributes, then I will try all kinds of combinations to better improve my model's accuracy. <br>
The second part I need to think about parameter settings is the `Multinomial Naive Bayes model`. At this point, I just use **grid search** method to figure the parameters out.

**Run the model** <br>
First, cd to the `ml` folder. <br>

Then, run:
```
python 2.MultinomialNB.py
```

It's very easy to run my model, there are basically two function. One is get_best_model(), you can run this function to get a best model and save it to best_model.pkl. It will also save the current best vectorizer into the local vectorizer.pkl. The second function is predict(). The first parameters are the sentence you want to convey. the second parameters are place where to load your model. So just run this two function. You can predict any sentence's sentiment.


**Evaluation Metric** <br>
Performance of our model:

               precision    recall  f1-score   support

           0       0.80      0.52      0.63       471
           1       0.76      0.90      0.82       430
           2       0.73      0.86      0.79       455
    accuracy                           0.76      1356

Also, for the accuracy, we got: <br>
Accuracy Score for trainning: 0.991519174041298. <br>
Accuracy Score for testing: 0.7566371681415929

The reason why there is a huge gap between training accuracy and testing accuracy is because sentence always has different meanings at different context. So without a specific context, a same sentence can be neutral, positive or negative. Even the testing accuracy is not high enough. It still can predict the accurate sentiment of one sentence. I tried many sentences for my model, and it also can give me the desired results.

**Sample prediction** <br>
We also generate some sample prediction using our model.
```
Sentence to be predicted: does I win? 
Predicted sentiment: neutral   

Sentence to be predicted: That doesn't concern me. 
Predicted sentiment: positive  

Sentence to be predicted: ONLY Western World economy WILL suffer.  
Predicted sentiment: negative  
```
Feel free to generate more by modifying our sentence to be predicted.

### 3. SVM
The aim of the Support Vector Machine algorithm is to identify a hyperplane within an N-dimensional space (where N represents the number of features) that effectively segregates the data points.SVM works well with sparse data, which is a common characteristic of text data when transformed into numerical representations.

**How?** <br>
By using the TF-IDF vectorizer, the data is transformed to a bag of word representation. SVM is effective in finding the optimal hyperplane that separates the different sentiment classes. <br>

**Parameter Setting** <br>
For the SVC model, I usedthe parameter kernel function  for transforming the data (e.g., 'linear', 'poly', 'rbf', or 'sigmoid'). 'C' the cost parameter. A smaller value creates a wider margin, which may result in more training errors but better generalization to the test data. Based on my own understanding toward the dataset, I changed the setting.
Another is TfidfVectorizer, I used that to tokenlize the test and remove default word.

**Run the model** <br>
First, cd to the `ml` folder. <br>

Then, run:
```
python 3.SVM.py
```

**Evaluation Metric** <br>
Performance of our model:

                  precision    recall  f1-score   support

              0       0.73      0.87      0.79       587
              1       0.83      0.15      0.26       162
              2       0.70      0.72      0.71       465
        accuracy                          0.72      1214
       macro avg      0.75      0.58      0.59      1214
    weighted avg      0.73      0.72      0.69      1214
|       | Predicted Negative | Predicted Neutral | Predicted Positive |
|-------|--------------------|-------------------|--------------------|
| True Negative  | 512              | 1                 | 74                 |
| True Neutral   | 68               | 25                | 69                 |
| True Positive  | 124              | 4                 | 337                |

