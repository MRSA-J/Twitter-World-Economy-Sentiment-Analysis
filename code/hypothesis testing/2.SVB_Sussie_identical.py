"""
Hypothesis: Before collapse or being bought, people's attitude towards SVB and Sussie Credit are identical
@ Author: ZhenHao Sun
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def hypothesis_testing(svb_file, credit_suisse_file):
    ### read csv files
    svb_df = pd.read_csv(svb_file)
    credit_suisse_df = pd.read_csv(credit_suisse_file)

    ### Clean the data, making sure there are no missing values or other issues:
    svb_df = svb_df.dropna()
    credit_suisse_df = credit_suisse_df.dropna()

    ### make sure they have roughly same number of row data
    svb_row = svb_df.shape[0]
    credit_suisse_row = credit_suisse_df.shape[0]
    assert abs(svb_row - credit_suisse_row) <= 100

    ### Extract the sentiment labels from both datasets:
    svb_sentiments = svb_df['label']
    credit_suisse_sentiments = credit_suisse_df['label']

    ### use the Chi-Square test, which is suitable for comparing categorical data:
    contingency_table = pd.crosstab(
        svb_sentiments,
        credit_suisse_sentiments,
        rownames=['SVB'],
        colnames=['Credit Suisse'],
        margins=True
    )

    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

    ### Evaluate the results. If the p-value is less than 0.05 (the significance level), we reject the null hypothesis (the attitudes are identical), otherwise, we fail to reject the null hypothesis:
    significance_level = 0.05

    if p_value < significance_level:
        print(
            f"Reject the null hypothesis. People's attitudes towards SVB and Credit Suisse are not identical. (p-value = {p_value})")
    else:
        print(
            f"Fail to reject the null hypothesis. People's attitudes towards SVB and Credit Suisse are identical. (p-value = {p_value})")


###############################################################################################################################
'''Main'''
###############################################################################################################################
svb_file = "../../data/after_clean/SVB Clean.csv"
credit_suisse_file = "../../data/after_clean/Credit Sussie Clean.csv"

hypothesis_testing(svb_file, credit_suisse_file)

"""
Conclusion: We Fail to reject the null hypothesis.
"""