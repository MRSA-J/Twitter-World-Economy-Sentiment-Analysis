"""

@ Author: ChenXi Wu
"""

import pandas as pd
import re
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import chardet
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Load the CSV file containing tweets
file_path = "../../data/world_march_labeled_cleaned.csv"

with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())

# Read the file with the detected encoding
df = pd.read_csv(file_path, encoding=result['encoding'])

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
df['Content'] = df['Content'].astype(str)

# Preprocessing
def preprocess(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return tokens

# Apply preprocessing on the 'text' column in the DataFrame
df['tokens'] = df['Content'].apply(preprocess)

# Create the Dictionary and Corpus needed for the LDA model
dictionary = Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

# Train the LDA model
num_topics = 5  # Change this value according to your requirements
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)

# Print the topics
topics = lda_model.print_topics()
for idx, topic in topics:
    print(f"Topic {idx}: {topic}")

# Save the model if you want to use it later
lda_model.save('lda_model.model')
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

#visualizate the model
pyLDAvis.save_html(vis_data, 'index.html')
