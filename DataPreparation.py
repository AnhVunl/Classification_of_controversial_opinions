## Import packages and load data

import pandas as pd
import numpy as np
import nltk
from nltk import tokenize, pos_tag, re, WordNetLemmatizer
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('punkt')
df = pd.read_csv('all_opinions.csv', encoding = 'utf-8', delimiter = ',')
# Transform 
df['target'] = df['per_curiam'].apply(lambda x: 1 if x == True else 0)
df = df[['text', 'target']]

# Text cleaning
# 1. stripping punctuations, numbers
# 2. lowercase
# 3. remove stopwords
# Remove stopwords, punctuation and pre-processing our reviews
def cleaning_text(review):
    # text cleaning steps
    only_letters = re.sub("[^a-zA-Z]", " ", review) # strip of symbols and punctuations
    stop_words = set(stopwords.words('english')) 
    tokens = nltk.word_tokenize(only_letters.lower()) # tokenize the reviews
    lemmatizer = WordNetLemmatizer() # lemmatize text
    lower_case = [l.lower() for l in tokens] #lower case all the words
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))#lower_case)) # keep tokens that are not stopwords
    words = []
    for w in filtered_result:
        if lemmatizer.lemmatize(w) not in words:
            words.append(w)
    return ' '.join(words)

df["clean_text"] = df["text"].apply(cleaning_text)
df.head()
# Save processed data 
df.to_csv(index = False)
