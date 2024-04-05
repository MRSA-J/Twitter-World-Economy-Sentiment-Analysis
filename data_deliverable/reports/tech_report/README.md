# Data Deliverable Tech Report

Your python version need to be `3.8 +` in order to run the `snscrape` webscraper.
If not, don't wory, the data we scrapped is already put in the data folder.

---

### A complete data spec

Below is the raw data which is web scrapped from twitter. We did add many data cleaning, selection and basic data analysis
using this data.

| Attribute     | Data Type | Default Value | Range/Domain                     | Unique | Required | Analysis Usage          | Sensitive Information  | Unique |
| ------------- | --------- | ------------- | -------------------------------- | ------ | -------- | ----------------------- | ---------------------- | ------ |
| `Tweet_id`    | Integer   | N/A           | N/A                              | Yes    | Yes      | Identify tweets         | No                     | Yes    |
| `User`        | String    | N/A           | N/A                              | No     | Yes      | User identification     | Yes (Handle with care) | No     |
| `Content`     | Text      | N/A           | N/A                              | No     | Yes      | Text analysis, NLP      | Yes (Handle with care) | No     |
| `Like_Count`  | Integer   | 0             | 0 to positive integers           | No     | Yes      | Engagement analysis     | No                     | No     |
| `Quote_Count` | Integer   | 0             | 0 to positive integers           | No     | Yes      | Engagement analysis     | No                     | No     |
| `Reply_Count` | Integer   | 0             | 0 to positive integers           | No     | Yes      | Engagement analysis     | No                     | No     |
| `Url`         | String    | N/A           | Valid Twitter status URLs        | Yes    | Yes      | Link to original tweet  | No                     | Yes    |
| `Date`        | Datetime  | N/A           | Timestamps within dataset bounds | No     | Yes      | Timeline-based analysis | No                     | No     |

#### Duplicate records

Actually there should be **no duplicate** records in our scrapped dataset. But just in case there are some that we didn't notice,
we can use `Twitter-id` to detect duplicate records directly.

If it’s not applicable, we will use a combination of `User`,`Content` and `Date` attributes to detect duplicates.
If we find two or more records with the same `User`, `Content` and very close or identical `Date` values, they are highly
likely to be duplicates.

Also, `URL` should be unique for each tweet, so we can probably use the url to detect duplicate content as well.

#### Intended analysis use for each data field

- `Tweet_id`: No, this field is for identification for the database only
- `User`: Yes, probably
- `Content`: Yes, definitely. We will do some cleaning on this field and separate the word and use the nltk to do some basic sentiment analysis
- `Like_Count`: Yes, probably (there should be an relationship with people's attitude and like_count
- `Quote_Count`: T.B.D similar to Like_Count. We wonder whether the Like Count alone is sufficient.
- `Reply_Count`: T.B.D similar to Like_Count. We wonder whether the Like Count alone is sufficient.
- `Url`: No
- `Date`: Yes, definitely, since we are studying people's attitute change according to the dates.

#### Sensitive information

We will carefully handle sensitive information like `User` and the `Content`, since we want to ensure user privacy.

For most of the cases, we couldn’t get the user's address, email number (those private information) through web scraping.
And, that is nice, since it prevents illegal use and prevents privacy issues naturally. <br>

But we will add some manual measures if needed. We can anonymize some identical information (name, address, email number)
if they are presented on our date. We can also try to use some encryption measures to protect users' data from unauthorized
access.

#### Required value

All of the scrapped value is not "required", in some ways. We could choose to only grab the info we need. But we do feel
like at least `User`, `Content`, `Date` is essential to our analysis.

#### Simplified analysis of the distribution of values

This does not apply to our data that much. However, it does apply to our data after we add the sentiment analysis columns
and we will illustrate this more in the next checkoff.

#### Link to the 100 rows of data in downloadable form (request by the TA even if we provided the full data downloadable form )

[100 rows sample data](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/data/sample%20data%20100%20rows)

#### Link to the full data in downloadable form

[data](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/tree/main/data)

---

### A sample of data (100 rows)

Our whole data folder can be easily opened so to make the readme concise, below will only show 5 rows of data to show the
format. And since our preprocessing method is similar for `SVB`, `Credit Sussie` and `World Economy` data, I will only show one
of them.

#### Raw data

| Tweet_id            | User         | Content                                                                                                                                                                                                                      | Like_Count | Quote_Count | Reply_Count | Url                                                         | Date                      |
| ------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------- | ----------- | ----------------------------------------------------------- | ------------------------- |
| 1634343371627170000 | LundSheridan | Wow this Vibecession is really heating up. Thanks @elonmusk for making systemic problems worse by enabling wild justifications because you have such delusional promotion and use wealth as a weapon for empty personal ego. | 0          | 0           | 0           | https://twitter.com/LundSheridan/status/1634343371627171840 | 2023-03-10 23:59:59+00:00 |
| 1634343367990440000 | KelleCorvin  | SVB collapse will have 'major' impact on tech industry                                                                                                                                                                       | 0          | 0           | 0           | https://twitter.com/KelleCorvin/status/1634343367990444033  | 2023-03-10 23:59:58+00:00 |
| 1634343363637020000 | Jannyleigh63 | @VivekGRamaswamy @SVB_Financial So they went the way of sri lanka!                                                                                                                                                           | 8          | 0           | 0           | https://twitter.com/Jannyleigh63/status/1634343363637022720 | 2023-03-10 23:59:57+00:00 |
| 1634343356477340000 | OhRebeccaO   | From a founder friend - the support group for SVB banked founders was 39 people this morning, and is now 400+                                                                                                                | 2          | 0           | 0           | https://twitter.com/OhRebeccaO/status/1634343356477349888   | 2023-03-10 23:59:55+00:00 |
| 1634343356472890000 | TheWuhanClan | @SVB_Financial Now watch who asks for a taxpayer funded bailout 馃槈                                                                                                                                                         | 16         | 0           | 0           | https://twitter.com/TheWuhanClan/status/1634343356472893440 | 2023-03-10 23:59:55+00:00 |

#### Data after adding sentiment, data cleaning and feature selection

| User | Content                                                                                                                                                                                                                      | Like_Count | Quote_Count | Reply_Count | Date      | content_words                                                                                                                              | words                                                                                                                                                                                                                                           | neg   | neu   | pos   | compound | label    |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------- | ----------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ----- | ----- | -------- | -------- |
| 2132 | Wow this Vibecession is really heating up. Thanks @elonmusk for making systemic problems worse by enabling wild justifications because you have such delusional promotion and use wealth as a weapon for empty personal ego. | 0          | 0           | 0           | 3/10/2023 | wow vibecession really heating thanks elonmusk making systemic problems worse enabling wild justifications delusional promotion use wealth | ['wow', 'vibecession', 'really', 'heating', 'thanks', 'elonmusk', 'making', 'systemic', 'problems', 'worse', 'enabling', 'wild', 'justifications', 'delusional', 'promotion', 'use', 'wealth', 'weapon', 'empty', 'personal', 'ego', 'bankrun'] | 0.28  | 0.429 | 0.291 | 0.3353   | positive |
| 1960 | SVB collapse will have 'major' impact on tech industry                                                                                                                                                                       | 0          | 0           | 0           | 3/10/2023 | collapse major impact tech industry daily mail online                                                                                      | ['collapse', 'major', 'impact', 'tech', 'industry', 'daily', 'mail', 'online']                                                                                                                                                                  | 0.314 | 0.686 | 0     | -0.4939  | negative |
| 1754 | @VivekGRamaswamy @SVB_Financial So they went the way of sri lanka!                                                                                                                                                           | 8          | 0           | 0           | 3/10/2023 | vivekgramaswamy svbfinancial went way sri lanka                                                                                            | ['vivekgramaswamy', 'svbfinancial', 'went', 'way', 'sri', 'lanka']                                                                                                                                                                              | 0     | 1     | 0     | 0        | neutral  |
| 2598 | From a founder friend - the support group for SVB banked founders was 39 people this morning, and is now 400+                                                                                                                | 2          | 0           | 0           | 3/10/2023 | founder friend support group banked founders people morning sad ground tough fundraise get caught mess beyond control                      | ['founder', 'friend', 'support', 'group', 'banked', 'founders', 'people', 'morning', 'sad', 'ground', 'tough', 'fundraise', 'get', 'caught', 'mess', 'beyond', 'control']                                                                       | 0.284 | 0.48  | 0.236 | -0.0516  | negative |
| 3568 | @SVB_Financial Now watch who asks for a taxpayer funded bailout 馃槈                                                                                                                                                         | 16         | 0           | 0           | 3/10/2023 | svbfinancial watch asks taxpayer funded bailout 馃槈                                                                                       | ['svbfinancial', 'watch', 'asks', 'taxpayer', 'funded', 'bailout', '馃槈']                                                                                                                                                                      | 0.219 | 0.781 | 0     | -0.1027  | negative |

#### How many?

There are **24160** data in total. <br>
To be specific, 10000 for SVB twitter data, 10000 for sussie credit twitter data, 4160 for world economy data (might consider adding this through editing the date)

The datasize is enough for our later hypothesis testing.

#### Identifying attributes

The identifying attributes are `Tweet_id`. Since `Tweet_id` is the unique identifier for each tweet and it’s a primary identifying attribute.
The unique URL is also unique for every record in the dataset. <br>

Notice that we are moving tweet id from the cleaned version of data, as our data aren't contain any duplicate and we are not using
database primary to do the sql analysis. So for the csv version, we simply remove it. And every row is 1 unique twitter data from our selected time range.

---

### Data Source

#### Reputable or not

We collected the data from the tweet, it’s **not reputable** source but tweet is a great database for us to carry out sentiment analysis.

#### How to generate the sample

We generate the sample through scraping the data with SVB, credit sussie, economics keywords. It’s comparably small (not that small) but representative data.  
When scraped the data, we are careful in handling the user information, since we wanted to ensure user’s privacy

#### Potential Bias

Our data might contain some bias, as not everyone uses twitter and there is a specific user distribution who would post
their thoughts more regarding the question. There are some people who don't use twitter as well. So although they might
have some thoughts with regard to the bank and economy we are studying, there is no way for us to analyze their attitutes.
Also, the date range of twitter we choose needs more careful consideration.

### Data Cleaning

We follow the following step to do the cleaning

1. Remove the null value (drop NA)
2. Lower all the words so that it is not case sensitive
3. Remove the punctuation using nltk package (i.e. `,` , `.`)
4. Remove the stop word using nltk package + the extra words we come up of
5. Split the words by " " and make a word list of all the twitter
6. Use the nltk package to analyze the sentiment and append negative, positive, neutral, compound score and label to our dataframe.
7. Plot 50 most frequent words for each dataset to give us a basic idea w.r.t the data.

**SVB March Plot**
![](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/plot/SVB%20March%20Twitter%2050%20common%20words.png)

**Credit Sussie March Plot**
![](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/plot/Credit%20Sussie%20March%20Twitter%2050%20common%20words.png)

**World Economy March Plot**
![](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/plot/World%20Economy%20March%20Twitter%2050%20common%20words.png)

### Other Data Distribution Plot

#### Twitter Date Distribution Plot

Since our SVB & Credit Sussie data only contains the day or 2 during the collapse and we are not using it to train the ML model,
we are only ploting the distribution for the world economy here.<br>

**World Economy March Plot**
![](https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/plot/World%20Twitter%20Date%20Distribution%20Plot.png)

#### Twitter Sentiment Distribution Plot

**World + SVB + Credit Sussie March Plot 1 (separate)**
![](<https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/plot/World_SVB_Sussie%20Twitter%20Sentiment%20Distribution%20Plot%20(separate).png>)

**World + SVB + Credit Sussie March Plot 2 (together)**
![](<https://github.com/MRSA-J/Twitter-World-Economy-Sentiment-Analysis/blob/main/plot/World_SVB_Sussie%20Twitter%20Sentiment%20Distribution%20Plot%20(together).png>)

### Challenges

1. **Special characters, emojis and incorrect formatting could affect data processing and analysis.** <br>
   We are working to remove the anomaly data columns
2. **Collecting and handling user information raises privacy concerns.**
3. **Our final hypothesis is still under discussion.** <br>
   Currently, our hypothesis is:
   1. Before collapse, 40% people have negative attitute towards SVB.
   2. Before collapse or being bought, people's attitute towards SVB and Sussie Credit are identical
   3. The collapse has an close relationship with people's attitute towards world economy. (The wording can be twisted)

### Notes after TA meeting

#### 1. Data Representative Issue

##### Description

Our TA thought that it would be better if we contain the data coming from the same time range.

Our TA also suggest that we could add a sampling method which samples the data proportionally to the date to make the
data more representative.

##### Solution & Our Attempt

We tried to expand our data (all 3 dataset) to contain the twitter from 3/01/2023 - 3/31/2023. But the package/api we use
becomes not free and cost much if we want to web scrapping ~10,000 data (cost $100+). This update w.r.t the package is made
around 3/31/2023. And right now other package also have some issue w.r.t the limitation of the data that it can scrape (even with
the twitter official api, it has this issue). This is because twitter update its authentication method and now it requires more
authentication if people want to scrape the data. So by setting this, they limit the amount of data which people can scrap. <br>

After discussion, we think that our data which we scrapped before is sufficient for the question we want to analysis, so we
don't want to waste our time and do the duplicative work to find another package, which meets our needs. (cuz we have already tried many)

To be specific, for our current data, both SVB and Credit Sussie twitter contains 10,000 data and our world economy data contains 4160 data.
It just exceed the data that we can scrape.

For the sampling method, since we are adjusting a little bit w.r.t our analysing strategy. We don't really need to do sample for
our data. (Or maybe consider doing it after I made the distribution plot.)

#### 2. Adding more Distribution Plot of Our Data

##### Description

Our TA mentioned that our original data distribution plot which describes the word frequency (excludes the stop word, special
word, and punctuation) in each dataset is not sufficient and think that we should add more data distribution plot which shows
whether our datas are 'screwed' or not b outliers.

#### Solution & Our Attempt

We add some distribution plot w.r.t our data (for example, the date of our twitter, and the sentiment distribution of our data)
to show that our data is not screwed.
