'''
@ Author MinFeiXue Zong
'''

import pandas as pd
from scipy.stats import ttest_ind

def hypo_testing(world_file, alpha=0.05):
    ### csv file import
    we_df = pd.read_csv(world_file)

    ### data cleaning
    we_df = we_df.dropna()
    
    ### Define the date of the collapse
    collapse_date = "2023-03-10"

    we_before = we_df[we_df["Date"] < collapse_date]
    we_after = we_df[we_df["Date"] >= collapse_date]

    ###conduct ttest&generate pvalue
    _, p_value = ttest_ind(we_before["compound"], we_after["compound"])


    if p_value < alpha:
        print("Reject the null hypothesis (H0). The attitudes towards the world economy are different before and after the collapse.")
    else:
        print("Fail to reject the null hypothesis (H0). The attitudes towards the world economy are identical before and after the collapse.")
    print(p_value)
    return p_value

world_data = "../../data/after_clean/World Economy Clean.csv"

hypo_testing(world_data)

