"""
Hypothesis: Before collapse, 50% people have negative attitute towards SVB.
@ Author: Chenxi Wu
"""

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Read the CSV file
svb_df = pd.read_csv('"../../data/after_clean/SVB Clean.csv"')

# Get the 'label' column
svb_sentiments = svb_df['label']

# Calculate the number of negative attitudes and total number of samples
negative_count = sum(svb_sentiments == 'negative')
total_count = len(svb_sentiments)

# Set the null hypothesis proportion
null_hypothesis_proportion = 0.5

# Perform the one-sample proportion test (using a one-sided alternative hypothesis)
stat, p_value = proportions_ztest(negative_count, total_count, null_hypothesis_proportion, alternative='larger')

# Set the significance level (alpha)
alpha = 0.05

# Print the results
print(f"Z-statistic: {stat:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < alpha:
    print("Reject the null hypothesis: The proportion of negative attitudes is significantly greater than 50%.")
else:
    print("Fail to reject the null hypothesis: There is insufficient evidence to conclude that the proportion of negative attitudes is greater than 50%.")
