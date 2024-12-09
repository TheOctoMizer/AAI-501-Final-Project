# -*- coding: utf-8 -*-
"""TF-IDF.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JrW7W3YfJhDb_cCJQ-Iixk3ulo-fNdj3
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('/kaggle/input/ai-vs-human-text/AI_Human.csv')

df.head()

"""# **Basic Information**"""

df.info()

df.describe()

sns.countplot(data=df,x='generated')

print('Total Texts:', df['generated'].count())
print('Human Written Texts:', (df['generated'] == 0.0).sum())
print('AI Generated Texts:', (df['generated'] == 1.0).sum())

"""# **Preprocessing**"""

df['text'][0]

def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')

    return text


df['text'] = df['text'].apply(remove_tags)

df['text'][0]

import string

string.punctuation

def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text

df['text']=df['text'].apply(remove_punc)

df['text'][0]

"""# **Spell Check**"""

import nltk
from nltk.corpus import words

nltk.download('words')
english_words = set(words.words())


def is_spelled_correctly(word):
    return word in english_words

word_to_check = df['text'][487232]
if is_spelled_correctly(word_to_check):
    print(f"The word '{word_to_check}' is spelled correctly.")
else:
    print(f"The word '{word_to_check}' is spelled incorrectly.")

df['text'][487232]

#import nltk
#from nltk.tok enize import word_tokenize, sent_tokenize
#from nltk.stem import PorterStemmer

#def correct_text(text):
 #   stemmer = PorterStemmer()
  #  english_words = set(words.words())
   # list_text = word_tokenize(text.lower())
    #stemmed_words = [stemmer.stem(word) for word in list_text]
    #for word in stemmed_words:
     #   if word not in english_words:
      #    return word

#correct_text(df['text'][0])

"""# **Stop Words Removal**"""

from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_words= ' '.join(filtered_words)
    return filtered_words

df['text']=df['text'].apply(remove_stopwords)

df['text'][0]

"""# **Splitting the Dataset**"""

y=df['generated']
X=df['text']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print(len(X_train))
print(len(y_train))

"""# **Pipeline**"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer()),  # Step 1: CountVectorizer
    ('tfidf_transformer', TfidfTransformer()),  # Step 2: TF-IDF Transformer
    ('naive_bayes', MultinomialNB())])

pipeline.fit(X_train, y_train)

y_pred= pipeline.predict(X_test)

"""# **Evaluation of Results**"""

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))