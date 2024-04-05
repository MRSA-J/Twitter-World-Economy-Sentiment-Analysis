
# Analysis Deliverable Tech Report
PS: This page is just for analysis deliverable tech report <br> 

A detailed hypothesis readme can be found at [here](https://github.com/csci1951a-spring-2023/final-project-ds-bang/tree/main/code/hypothesis%20testing) <br>
A detailed ML readme can be found at [here](https://github.com/csci1951a-spring-2023/final-project-ds-bang/tree/main/code/ml) <br>

If you are interested in running our code, feel free to check it out.

### Hypothesis Testing
#### Our hypothesis
1. Before collapse, 30%/40%/50% (in code, 50%, but we have tests all of them) people have negative attitute towards SVB.
2. Before collapse or being bought, people's attitude towards SVB and Sussie Credit are identical.
3. The collapse has an close relationship with people's attitute towards world economy. 

### 1.Before collapse, 50% people have negative attitute towards SVB.

##### 1.1 Why did you use this statistical test or ML algorithm? Which other tests did you consider or evaluate? What metric(s) did you use to measure success or failure, and why did you use it? What challenges did you face evaluating the model?Did you have to clean or restructure your data?  
To test if 50% of people had a negative attitude towards SVB, we use a one-sample z-test for proportions. This test will help us determine if the proportion of negative attitudes in our sample (from the 4000 tweets) is significantly different from the hypothesized proportion of 0.5 (50%). <br>

We chose this test because:
1. We have a large sample size of 4,000 tweets, which satisfies the requirements for the Central Limit Theorem.
2. We are comparing a sample proportion (proportion of negative attitudes in your dataset) to a hypothesized proportion (50%).

##### 1.2 Which other tests did you consider or evaluate?
There are no other tests that can better fit the scenarios. The one-sample t-test is not appropriate for this case, as it is designed to compare the mean of a continuous variable in a single sample to a known population mean or a hypothesized value. In this scenario, we are working with proportions 
(i.e., the proportion of people with a negative attitude towards SVB) which are categorical in nature, not continuous.

##### 1.3 What metric(s) did you use to measure success or failure, and why did you use it?
For measuring success or failure in this context, the p-value is the key metric. It helps determine if there is a statistically significant difference between the sample proportion and the hypothesized proportion. We compare the p-value to a significance level (0.05 in our case) to decide whether to 
reject or accept the null hypothesis.

##### 1.4 What challenges did you face evaluating the model?
Some challenges we faced include noisy data, mislabeled sentiments, or non-representative samples. Because the sentiments are labeled using a ML model, it could be wrong or misinterpreted in some cases. Moreover, it is difficult to evaluate how representative of the population is due to the constraints set by Twitter.

##### 1.5 Did you have to clean or restructure your data?
Before conducting the test, we preprocessed your data to ensure it is in the correct format. This has been clearly described in the data cleaning section of the report. Once the data is cleaned and structured, we can proceed with the statistical test.

##### 1.6 What is your interpretation of the results? Do you accept or deny the hypothesis, or are you satisfied with your prediction accuracy? For prediction projects, we expect you to argue why you got the accuracy/success metric you have. Intuitively, how do you react to the results? Are you confident in the results?
We have a p-value of 1.00 and a z-statistic of -22.981 for testing the hypothesis that the proportion of negative attitudes towards SVB is larger than 50%. The results suggest that we fail to reject the null hypothesis. Less than 50% of the tweets in your sample have a negative attitude towards SVB. Since the z-statistic 
is negative, it means that the sample proportion of negative attitudes is less than the hypothesized proportion of 50%. <br>

When we change the hypothesized proportion to 30% and obtain a z-statistic of 17.153 and a p-value of 0.000, it means there is strong evidence to reject the null hypothesis. We can conclude that the proportion of negative attitudes towards SVB in your sample is significantly larger than 30%. Since the z-statistic is positive, 
it suggests that the sample proportion is greater than the hypothesized proportion, meaning that more than 30% of the tweets in your sample have a negative attitude towards SVB. <br>

Based on the results provided, we fail to reject the hypothesis that 50% of the tweets have a negative towards SVB before collapse. But we can confidently reject the null hypothesis when comparing the sample proportion to a hypothesized proportion of 30%. This suggests that more than 30% of the tweets in your sample have a 
negative attitude towards SVB. These results match with our expectation as not many people have negative attitudes towards SVB before it collapsed.

##### 1.7 Provide comments and an interpretation of the results you obtained. 
##### 1.7.1 Did you find the results corresponded with your initial belief in the data? If yes/no, why do you think this was the case?
The results align with our initial belief. Because our data is relatively large and representative of the day before SVB collapsed. Moreover, the results indicate a significant difference between the sample proportion of negative attitudes towards SVB and the hypothesized proportions (both 50% and 30%).

##### 1.7.2 Do you believe the tools for analysis that you chose were appropriate? If yes/no, why or what method could have been used?
The one-sample z-test for proportions was chosen as it is appropriate for testing the difference between a sample proportion (proportion of negative attitudes in our dataset) and a hypothesized proportion (e.g., 50% or 30%). The test is suitable for large sample sizes and proportions, which is the case in our dataset.

##### 1.7.3 Was the data adequate for your analysis? If not, what aspects of the data were problematic and how could you have remedied that?
We try our best to ensure the sentiment labels are accurate, and consider the representativeness of our sample to the larger population. However, we are not generalizing our findings to a broader timeframe. Due to technical constraints set by Twitter, our data is collected on the data before it collapses. Thus, our model 
does not generalize for a longer period of time.


### 2. Before collapse or being bought, people's attitude towards SVB and Sussie Credit are identical.

##### 2.1 Why did you use this statistical test or ML algorithm? Which other tests did you consider or evaluate? What metric(s) did you use to measure success or failure, and why did you use it? What challenges did you face evaluating the model? Did you have to clean or restructure your data?
The Chi-Square test would be a suitable choice. This test is used for comparing categorical variables, which fits the context of sentiment analysis (positive, negative, and neutral sentiment labels). It helps to determine if there is a significant difference in the distribution of sentiments between the two groups (SVB and Credit Suisse). 

Other tests that could be considered for comparing categorical variables include the Fisher's Exact Test, which is suitable for small sample sizes, or the G-test, which is similar to the Chi-Square test but uses the likelihood ratio instead of the sum of squared differences. <br>

 In the context of hypothesis testing, the primary metric used is the p-value. The p-value helps to determine if the observed differences in sentiment distributions are due to chance or if there is a significant relationship between the sentiments and the two groups (SVB and Credit Suisse). <br>

The challenge I mainly deal with is ensuring that the assumptions of the chosen test are met, such as having a large enough sample size for the Chi-Square test, and having mutually exclusive and independent categories. <br>

Yes, the reason is ensuring that the sentiment labels are consistent across both datasets (SVB and Credit Suisse).

##### 2.2 What is your interpretation of the results? Do you accept or deny the hypothesis, or are you satisfied with your prediction accuracy? For prediction projects, we expect you to argue why you got the accuracy/success metric you have. Intuitively, how do you react to the results? Are you confident in the results?
The interpretation of my result relies on the p-value. I accept my hypothesis, and I feel satisfied with the results, since it’s consistent with the real circumstances.

##### 2.3 Provide comments and an interpretation of the results you obtained:
##### 2.3.1 Did you find the results corresponded with your initial belief in the data? If yes/no, why do you think this was the case?
Yes, even though we crawl the data before collapse, we can easily see that people are feeling anxious about the current economy. Hence, if something bad will happen, there must be some tips we can see before.

##### 2.3.2 Do you believe the tools for analysis that you chose were appropriate? If yes/no, why or what method could have been used?
Yes, since we collect the data randomly and use the most-accurate model to label each sentence. After that we even label the data again based on our own understanding. So we can assure the dataset is very convincing. By using the hypothesis testing method, we can easily see a trend behind these data, and that is the 
accurate prediction for the future.

##### 2.3.3 Was the data adequate for your analysis? If not, what aspects of the data were problematic and how could you have remedied that?
We think the data is adequate for our analysis, since there might be so many new tweets appearing online. We cannot collect all of them, but we can take a large part of it and use that to analyze our hypothesis. This is the very famous method often used in statistics, and It is tested through thousands of years.

### 3.The collapse has an close relationship with people's attitute towards world economy. (The wording can be twisted).

##### 3.1 Why did you use this statistical test or ML algorithm? Which other tests did you consider or evaluate? What metric(s) did you use to measure success or failure, and why did you use it? What challenges did you face evaluating the model? Did you have to clean or restructure your data?

The challenge I face is assumption, it’s hard to figure out whether iI chose this h0 because I’m curious about people’s attitude toward the economy before and after collapse.I used the statistical test since it’s easier and I believed t test is ideal for this analysis because:
1. We have two independent samples 
2. We are comparing the means
3. I assume that the data in each group are normally distributed. I used the p-value and set the significance level as 0.05, then I evaluated the p-value against the significance level.t’s normally distributed from a broad perspective, I may try to find some papers to support my assumption

We had cleaned my data by dropping na values.


We defined our null hypothesis as "There is no significant difference in the mean sentiment scores between the group observed before the economic collapse and the group observed after the economic collapse." This hypothesis implies that the economic event (the collapse) did not significantly alter public sentiment towards the economy.

##### 3.2 What is your interpretation of the results? Do you accept or deny the hypothesis, or are you satisfied with your prediction accuracy? For prediction projects, we expect you to argue why you got the accuracy/success metric you have. Intuitively, how do you react to the results? Are you confident in the results?
There is a statistically significant difference in the attitudes towards the world economy before and after the collapse: the p-value of 0.04 less than 0.05 would lead us to reject the null hypothesis.

##### 3.3 Provide comments and an interpretation of the results you obtained:
##### 3.3.1 Did you find the results corresponded with your initial belief in the data? If yes/no, why do you think this was the case?  
Yes, I found the result correspond with the initial belief. To the common sense, the collapse may have some impact on the people’s attitude towards the economy. 
##### 3.3.2 Do you believe the tools for analysis that you chose were appropriate? If yes/no, why or what method could have been used?
I guess it may be because of my assumption that people’s attitudes towards the economy are normally distributed. I guess we should consider alternative statistical tests, such as the Mann-Whitney U test (a non-parametric test), which does not require the data to be 
normally distributed and can be used to compare the medians of two groups instead.
##### 3.3.3 Was the data adequate for your analysis? If not, what aspects of the data were problematic and how could you have remedied that?
Due to technical restrictions imposed by Twitter, our data is collected only before the collapse. As a result, our model's applicability over longer periods is constrained. Next time we may try to gather the data from somewhere without those constraints.


### ML
#### Our ML
1. LDA
2. Multinomial Naive Bayes
3. SVM

### 1. LDA
##### 1.1 Why did you use this statistical test or ML algorithm?
We use Latent Dirichlet Allocation (LDA) algorithm because it 
is a widely-used unsupervised machine learning technique for topic modeling. LDA helps discover the hidden thematic structure in a collection of documents, making it suitable for analyzing and summarizing large text corpora.

##### 1.2 Which other tests did you consider or evaluate?
There are alternative topic modeling algorithms that could be considered: <br>
1. Non-negative Matrix Factorization (NMF): Another unsupervised learning method that decomposes a matrix into two lower-dimensional matrices. It can be used for topic modeling by decomposing a term-document matrix.
2. Latent Semantic Analysis (LSA): An unsupervised method that applies singular value decomposition (SVD) to a term-document matrix to reduce its dimensions, revealing the latent semantic relationships between terms and documents.

##### 1.3 What metric(s) did you use to measure success or failure, and why did you use it?
We evaluate the model from two important perspectives. First is topic coherence: A measure of the semantic similarity between the top terms within a topic. Higher coherence generally corresponds to more interpretable topics. Second is interpretability. 
Human interpretation and judgment of the generated topics' quality and interpretability, especially in terms of understanding the underlying themes in the data.

##### 1.4 What challenges did you face evaluating the model? Did you have to clean or restructure your data?
Here are some challenges in evaluating topic models include: <BR>
1. Interpretability: Ensuring that the topics generated by the model are meaningful and easily interpretable by humans.
2. Choosing the hyperparameters (the optimal number of topics): Selecting the right number of topics to balance granularity and interpretability.
3. Model evaluation: Relying on human judgment for evaluation can be subjective.

##### 1.5 Did you have to clean or restructure your data?
Our preprocessing steps include: <br>
1. Lowercasing: Converting all text to lowercase to ensure consistency.
2. Tokenization: Splitting the text into individual words or tokens.
3. Stopword removal: Removing common words (e.g., "the," "and," "is") that do not provide meaningful information about the content.
4. Lemmatization: Converting words to their base forms (e.g., "running" to "run") to reduce the feature space and improve model performance.

##### 1.6 What is your interpretation of the results? Do you accept or deny the hypothesis, or are you satisfied with your prediction accuracy? For prediction projects, we expect you to argue why you got the accuracy/success metric you have.Intuitively, how do you react to the results? Are you confident in the results?
Based on our results, the top first topic is on how the war affects the world economy. For example, “war”, “ukraine”, “world economy”, “collapse”, and “debt” are highly correlated. The second common topic is closely related to the first one. But it focuses more on international relationships and the world economy. 
For example, “ukraine”, “russia”, “china”, “countries”, “global”, and “status” are frequently mentioned together. The third topic is more about monetary policies and the world economy. Because the key words are “inflation”, “federal reserve”, “market”, “inflation”, “business”, and “interest”. Topic 4 and 5 are less 
well-established but they basically give us information about people’s different attitudes towards the world economy. Some people are looking for “change”, some people think it is a “failure”, and some people think we need more “control”. We are satisfied with the model prediction. <br>

###### a visualization can also be found [here](https://chenxiwub.github.io/ldaplot/#topic=0&lambda=1&term=)

We think our model is accurate and successful because we have manually labeled the content from Twitter for supervised learning. The topic summarized by our LDA model is consistent with our human observations. Intuitively, we think these are highly popular topics on the world economy and confidence about our results. 

##### 1.7 Provide comments and an interpretation of the results you obtained:
##### 1.7.1 Did you find the results corresponded with your initial belief in the data? If yes/no, why do you think this was the case?
The topics the model identified are coherent and seem to cover distinct aspects of the world economy and its relationship with war, international relationships, and monetary policies. It's a positive sign that the topics generated by the LDA model are consistent with our manual observations and expectations. 
This implies that the model has effectively learned the underlying thematic structure in the dataset. The LDA model has performed well in this particular use case and aligns with our initial belief in the data.

##### 1.7.2 Do you believe the tools for analysis that you chose were appropriate? If yes/no, why or what method could have been used?
The LDA (Latent Dirichlet Allocation) algorithm is an appropriate choice for topic modeling in this case, especially when we have a large text corpus from Twitter and want to discover the hidden thematic structure in it. LDA is a popular and widely-used unsupervised machine learning technique that has proven effective 
in a variety of text analysis tasks.LDA is an unsupervised learning, which does not require labeled data, making it suitable for exploring and summarizing a text corpus without prior knowledge of its content. LDA generates human-readable topics, which can provide insights into the main themes present in the dataset. There are other methods such as Non-negative Matrix Factorization (NMF) and Latent Semantic Analysis (LSA) that can be used and described above.

##### 1.7.3 Was the data adequate for your analysis? If not, what aspects of the data were problematic and how could you have remedied that?
The data is adequate because it is a large size data. A larger dataset usually provides better results, as it allows the topic modeling algorithm to learn more about the underlying thematic structure. Data is of high quality as the data are clean and well-structured, without excessive noise, missing values, or irrelevant information. Moreover, the dataset is a diverse dataset, containing a wide range of themes and perspectives from people with different backgrounds, 
and it can help the topic modeling algorithm learn a richer and more accurate representation of the underlying thematic structure.  <br>

Some potential improvements include: <br>

1.Consider incorporating additional data sources to ensure that the full range of themes and perspectives is represented. 
2.Identify and mitigate any biases in the data collection, sampling, or preprocessing procedures to ensure that the dataset is unbiased and suitable for analysis.

### 2. Multinomial Naive Bayes
##### 2.1 Why did you use this statistical test or ML algorithm? Which other tests did you consider or evaluate? What metric(s) did you use to measure success or failure, and why did you use it? What challenges did you face evaluating the model? Did you have to clean or restructure your data?
We used the Multinomial Naive Bayes algorithm because it is a simple and effective probabilistic classifier suitable for text classification tasks, such as sentiment analysis. It works well with discrete data, such as the word frequencies obtained from the text through the TF-IDF vectorization process. <br>

We also considered using Logistic Regression, another widely-used classification algorithm that works well for binary and multi-class problems. It is a linear model that estimates the probability of an observation belonging to a particular class based on a linear combination of input features. Other algorithms that could be considered for text classification tasks include Support Vector Machines, Decision Trees, Random Forests, and neural networks such as LSTM or Transformer-based models. <br>

We used accuracy as a primary metric to measure the success of the model. Accuracy is the proportion of correct predictions out of the total predictions made, and it provides a simple way to evaluate the overall performance of a classifier. However, it is important to note that accuracy may not be the best metric if the dataset is imbalanced. In such cases, precision, recall, and F1-score can be more appropriate metrics for assessing the model's performance on different classes.

Some challenges we have includes **the way to deal with the imbalanced datasets**. In this case, we used the Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset before training the model. This can help improve the model's performance on underrepresented classes. <br>

We also have to reconstruct our data. The method we perform includes:
1. Preprocessing the text by removing special characters, converting text to lowercase, eliminating stopwords, and lemmatizing words.
2. Converting the preprocessed text into a numerical format using the TF-IDF vectorizer.
3. Balancing the dataset using SMOTE to address the class imbalance issue.
4. Splitting the dataset into training and testing sets for model evaluation.

##### 2.2 What is your interpretation of the results? Do you accept or deny the hypothesis, or are you satisfied with your prediction accuracy? For prediction projects, we expect you to argue why you got the accuracy/success metric you have. Intuitively, how do you react to the results? Are you confident in the results?
My interpretation of the results is that the model achieved moderate success in predicting the sentiment of the tweets. With an accuracy of around 75%, there is certainly room for improvement. However, given the simplicity of the Multinomial Naive Bayes model and the inherent complexity and subjectivity of sentiment analysis, this level of accuracy can be considered acceptable for a baseline model. <br>
Intuitively, given the complexity of sentiment analysis and the challenges in representing and modeling text data, the obtained results are reasonable but not exceptional. There is room for improvement, and experimenting with more advanced algorithms, feature extraction techniques, and better handling of imbalanced data could lead to better performance. I am actually confident with the results. Since the sentiment analysis model would always be subjective. I think without the specific context, 
nobody can really predict the actual sentiment of the sentence.

##### 2.3 Provide comments and an interpretation of the results you obtained:
##### 2.3.1 Did you find the results corresponded with your initial belief in the data? If yes/no, why do you think this was the case?  
The results partially corresponded with my initial belief in the data. Given the simplicity of the Multinomial Naive Bayes algorithm and the complexities of sentiment analysis, I expected moderate performance from the model. The achieved accuracy of around 75% aligns with this expectation. However, there is room for improvement, and I believe that more advanced models could achieve better performance on this task. The tools chosen for the analysis were appropriate for an initial baseline model. 
##### 2.3.2 Do you believe the tools for analysis that you chose were appropriate? If yes/no, why or what method could have been used?
The Multinomial Naive Bayes classifier is known for its simplicity and ease of implementation, making it a good starting point for a text classification task like sentiment analysis. However, there are more advanced techniques and models available, such as deep learning-based models, which may offer improved performance in capturing complex language patterns and semantics.
##### 2.3.3 Was the data adequate for your analysis? If not, what aspects of the data were problematic and how could you have remedied that?
The feature extraction method, TF-IDF vectorization, is a commonly used technique for representing text data in a machine learning model. However, more advanced techniques like word embeddings or contextual embeddings from pre-trained language models could potentially lead to better results. The data provided for the analysis was adequate, but there were some limitations, such as the class imbalance in the dataset. To address this issue, we used the SMOTE technique to balance the dataset. However, the synthetic samples generated might not perfectly represent the actual distribution of the minority classes, which could affect the model's ability to generalize well to new data.

### 3. SVM
##### 3.1 Why did you use this statistical test or ML algorithm? Which other tests did you consider or evaluate? What metric(s) did you use to measure success or failure, and why did you use it? What challenges did you face evaluating the model? Did you have to clean or restructure your data?
The Support Vector Machine algorithm is used since SVM is effective in finding the optimal hyperplane that separates the different sentiment classes. I also considered Naive Bayes and Linear Regression. We can use the models like precision, recall, F1 score to measure the success of the model.These metrics are suitable for evaluating the performance of a classification model, especially when the classes are imbalanced. <br>

The challenges we faced are:
1. Sentiment can be subjective, and some text samples may be challenging to classify even for humans 
2. SVM may not be quite suitable for large datasets, it may not perform well when the dataset has more noise. <br>

For the data cleaning and restructing, we performed the following:
1. Lowercase the text
2. Remove stop words and punctuation
3. Lemmatize the word
4. Use SentimentIntensityAnalyzer from the NLTK to calculate the sentiment scores
5. Label the sentiment base on 4)
6. Use the TfidfVectorizer to transform the test in to numerical format.

##### 3.2 What is your interpretation of the results? Do you accept or deny the hypothesis, or are you satisfied with your prediction accuracy? For prediction projects, we expect you to argue why you got the accuracy/success metric you have. Intuitively, how do you react to the results? Are you confident in the results?
The results are as follows: <br>
Classification report

| Sentiment | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.73      | 0.87   | 0.79     | 386     |
| Neutral   | 0.75      | 0.20   | 0.32     | 103     |
| Positive  | 0.73      | 0.73   | 0.73     | 320     |
| Accuracy  |           |        | 0.73     | 809     |
| Macro Avg | 0.74      | 0.60   | 0.62     | 809     |
| Weighted Avg | 0.73   | 0.73   | 0.71     | 809     |

Confusion matrix

|       | Predicted Negative | Predicted Neutral | Predicted Positive |
|-------|--------------------|-------------------|--------------------|
| True Negative  | 336              | 1                 | 49                 |
| True Neutral   | 45               | 21                | 37                 |
| True Positive  | 80               | 6                 | 234                |

The SVM model achieved an accuracy of around 0.72 for sentiment classification. The model performs best at classifying negative sentiment, with an F1-score of 0.79. It still struggles with the neutral class, with an F1-score of only 0.32. This might be due to the limited number of neutral examples in the dataset or the inherent ambiguity in neutral sentiment. I’m satisfied with the 
result since sentiment analysis is hard even for humans.  <br>

I think the prediction accuracy should be improved, so I want to explore other methods like fine-tuning hyperparameters, using more advanced techniques like deep learning, or oversampling the underrepresented classes. The model achieved an accuracy of 0.72, i used a series of text preprocessing techniques, such as lowercasing, removing stop words, and lemmatization. 
These steps help reduce noise in the text and create a cleaner input for the machine learning algorithm. And the TfidfVectorizer is used to transform the text into numerical features. Also chose SVM with a linear kernel as our classifier and used a 70-30 train-test split, which provided an appropriate balance between training data for model building and testing data for evaluation. <br>

There is room for improvement, particularly in the classification of neutral sentiment, which had low recall and F1-score. I may adopt some improvements including: different feature extraction techniques, such as word embeddings and try to adopt hyperparameters using techniques like grid search or random search.

##### 3.3 Provide comments and an interpretation of the results you obtained:
##### 3.3.1 Did you find the results corresponded with your initial belief in the data? If yes/no, why do you think this was the case?  
I find the result correspond with my initial belief, the sentiment analysis is a well-studied problem and the preprocessing techniques, feature extraction methods, and model selection used were appropriate for the task at hand.
##### 3.3.2 Do you believe the tools for analysis that you chose were appropriate? If yes/no, why or what method could have been used?
The preprocessing steps, TfidfVectorizer, and the linear SVM classifier are appropriate for analysis, since those techniques are commonly used in text classification tasks, including sentiment analysis. But I will try to explore additional techniques and refinements that could lead to improved performance.
##### 3.3.3 Was the data adequate for your analysis? If not, what aspects of the data were problematic and how could you have remedied that?
We might need a larger dataset to train a more robust model. The distribution of the classes might be uneven, leading to biases in the model's predictions. To remedy this, techniques such as oversampling or undersampling can be employed. There exists some ​​ambiguity in the sentiment labels: Some text samples might be hard to classify due to their ambiguous or context-dependent nature. 
So further refinement of the data may be necessary.
