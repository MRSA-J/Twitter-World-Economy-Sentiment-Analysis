# Final-Project-Ds-Bang

Your python version need to be `3.8 +` in order to run the `snscrape` webscraper.
If not, don't wory, the data we scrapped is already put in the data folder.

---

### Project Content

- Data Deliverable (Checkoff 2)

  - [Sample Data Download Link (100 rows for all the csvs)](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/data/sample%20data%20100%20rows)
  - [Full Data Download Link](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/data)
  - [Data Concise Tech Report](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/data_deliverable/reports/tech_report)

- Analysis Deliverable (Checkoff 3)

  - [ML Component Readme](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/code/ml)
  - [Hypothesis Testing component Readme](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/code/hypothesis%20testing)
  - [Analysis Report](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/analysis_deliverable/reports)

- Final Deliverable
  - [Visualizations](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/final_deliverable/visualizations)
  - [Abstract](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/final_deliverable/abstract.pdf)
  - [Socio-historical context and impact report](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/final_deliverable/Socio-historical%20context%20and%20impact%20report.pdf)
  - [Interactive Component](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/final_deliverable/interactive_component/Bang)
  - [Poster](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/final_deliverable/DS%20Bang%20Poster.pdf)
  - [Presentation](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/final_deliverable/DS%20project%20presentation.mp4)

### File Table

| Function name                                  | Description                                                                                                                       |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `webscrap.py`                                  | The script to webscrap the data from twitter.                                                                                     |
| `preprocessing.py`                             | The script to do the basic data cleaning and generate some plot w.r.t the data                                                    |
| `labels_cleaning.ipynb`                        | Calculate the overall sentiment label based on our manual labeling, drop/deal with the abnormal data, show the label distribution |
| `ml/1.LDA.py`                                  | Our ML model 1 LDA                                                                                                                |
| `ml/2.MultinomialNB.py`                        | Our ML model 2 Multinomial Naive Bayes                                                                                            |
| `ml/3.SVM.py`                                  | Our ML model 3 SVM                                                                                                                |
| `hypothesis_testing/1.50_negative.py`          | Our Hypothesis testing 1                                                                                                          |
| `hypothesis_testing/2.SVB_Sussie_identical.py` | Our Hypothesis testing 2                                                                                                          |
| `hypothesis_testing/3.Collapse_World.py`       | Our Hypothesis testing model 3                                                                                                    |

---

### Todo

- [x] Check off 0 (Proposal)
- [x] Check off 1 (Data Deliverable)
  - [x] Data Spec
  - [x] Tech report w.r.t the data
  - [x] Meet with mentor TA
- [x] Check off 2 (Analysis Deliverable)
  - [x] Hypothesis Testing Component
  - [x] ML Component
  - [x] Analysis report
  - [x] Meet with mentor TA
- [x] Final submission
  - [x] Interactive component

---

### Contribution

#### Contributor

[@Chen Wei](https://github.com/MRSA-J).
[@MinFeiXue Zong](https://github.com/SereneZong).
[@ChenXi Wu](https://github.com/ChenxiwuB).
[@ZhenHao Sun](https://github.com/Asher-Sunzh).

#### Detailed Contribution

- Brainstorming the idea of the project & Write Project Proposal: Together
- Collecting and Label the data
  - Webscraping the data: Chen Wei
  - Manually Label the Data: MinFeiXue Zong, ChenXi Wu, ZhenHao Sun
  - Organize the data & Generalize the Overall Sentiment Label: Chen Wei
  - Drop the Strange / Abnormal Rows: Chen Wei
- Preprocessing the data
  - Basic Data Cleaning: Chen Wei
  - Twitter Date Distribution Plot: Chen Wei
  - Twitter Sentiment Distribution Plot: Chen Wei
  - Generate word frequency plot: Chen Wei
  - Data Deliverable Tech Report: Chen Wei
  - Use pretrained model to label SVB/Credit Sussie Data: Chen Wei
- Analysis
  - Add extra feature (column) of the data: Together
  - Hypothesis Testing
    1.  Before collapse, 30%/40%/50% (in code, 50%, but we have tests all of them) people have negative attitute towards SVB: ChenXi Wu
    2.  Before collapse or being bought, people's attitude towards SVB and Sussie Credit are identical: ZhenHao Sun
    3.  The collapse has an close relationship with people's attitute towards world economy. (The wording can be twisted): MinFeiXue Zong
  - ML Component
    1.  LDA (Unsupervised Learning Method): ChenXi Wu
    2.  Multinomial Naive Bayes: ZhenHao Sun
    3.  SVM: MinFeiXue Zong
  - Organize the Code and Help with Debugging: Chen Wei
  - Analysis Report: Together
- Interactive component: Zhenhao Sun
- Socio & Ethical report: Minfeixue Zong

---

## License

[MIT](LICENSE)
