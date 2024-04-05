
# Hypothesis Testing


----
### 1.

**Hypothesis:** Before collapse, more than 50% of people had a negative attitude towards SVB.

**Dataset:** Dataset `SVB clean.csv` are crawled online before the day that the bank collapse. 

**The label category:** 
The two dataset both contains a sentiment analysis label, which have the value `0, 1, 2`. <br>
```
0. negative
1. positive
2. neutral
```
The number of negative attitudes is calculated by counting the number of samples with a 'negative' sentiment label, and the total number of samples is also calculated.

**How to run the model**:
First, cd to the `hypothesis testing` folder. <br>

Then run
```
python 1.50_negative.py
```

**Result:** 
Fail to reject the null hypothesis. Based on the results provided, we fail to reject the hypothesis that 50% of the tweets have a negative towards SVB before collapse. (p-value = 1.00).

To be specific, our p-value: 1.00 > 0.05. 
That is to say we fail to reject the hypothesis. We can conclude that the proportion of negative attitudes towards SVB before collapse in our sample is not significantly larger than 50%. 

### 2.
**Hypothesis:** Before collapse or being bought, people's attitude towards SVB and Sussie Credit are identical.

**Dataset:** Both dataset, `Credit Sussie clean.csv` and `SVB clean.csv` are crawled online before the day that the bank collapse/being bought. 
Hence, I can compare these two dataset's context label to get a valid proof for my hypothesis testing

**The label category:** 
The two dataset both contains a sentiment analysis label, which have the value `0, 1, 2`. <br>
```
0. negative
1. positive
2. neutral
```
So we create the **contingency table** using the pandas crosstab() function. <br>

PS: The contingency table is a tabular representation of the relationship between two categorical variables (in this case, the sentiment labels for SVB and Credit Suisse). 
This table is used as input for the Chi-Square test to determine if there's a significant difference in the distribution of sentiment labels between the two groups.

**How to run the model**:
First, cd to the `hypothesis testing` folder. <br>

Then run
```
python 2.SVB_Sussie_identical.py
```

**Result:** 
Fail to reject the null hypothesis. People's attitudes towards SVB and Credit Sussie are identical. (p-value = 0.9629995589325446).

To be specific, our p-value: 0.9629995589325446 > 0.05. 
That is to say we fail to reject the hypothesis. Therefore, By looking at the distribution of these two datasets,
we can say that before collapse, people's attitude towards SVB and Sussie Credit are almost identical



### 3. 
**Hypothesis:** The collapse has an close relationship with people's attitute towards world economy. (The wording can be twisted) 

**Dataset:** The dataset 'World Economy clean.csv'. It contains data related to public sentiment towards the world economy, derived from social media posts.

**The label category:** 
The dataset includes a sentiment analysis label, which is assigned one of three possible values: 0, 1, or 2. These values represent the overall sentiment expressed in each post(like the label system in Hypothesis 2)
The dataset includes a 'compound' score, which represents the overall sentiment expressed in each post. This score is a continuous value ranging from -1 (most negative sentiment) to 1 (most positive sentiment). The compound score serves as the label category, allowing us to compare and analyze sentiment before and after the economic collapse.

**How to run the model:**
First, cd to the `hypothesis testing` folder. 

Then run
```
python 3.Collapse_World.py
```

**Result:** 
The p-value is  0.0400179627, which is smaller than the significance level 0.05. We reject the null hypothesis that people have the same attitudes towards the world economy before and after the collapse. The average sentiment scores from Twitter towards the world economy are statistically different before and after the collapse.
