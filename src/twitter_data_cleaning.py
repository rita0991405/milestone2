import pandas as pd
import gzip
import json
import gdown
from google.colab import drive
drive.mount('/content/drive')

import os

folder_path = '/content/Twitter'

if os.path.isdir(folder_path):
    print(f"Listing files in: {folder_path}")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if files:
        print("Found files:")
        for f in files:
            print(f"- {f}")

all_dfs = []

for file_name in files:
    if file_name.endswith('.tsv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, sep='\t')
        all_dfs.append(df)

first_df = pd.concat(all_dfs, ignore_index=True)

print(f"Combined dataframe has {len(first_df)} rows and {len(first_df.columns)} columns.")
first_df.head()

"""Load all files from the first Twitter dataset and concatenate them into a single dataframe."""

first_df.info()

"""Drop the class label column."""

first_df = first_df.drop(columns='class_label')
print(first_df.head())

"""Load and combine with the second Twitter dataset"""

file = 'training.1600000.processed.noemoticon.csv'
second_df = pd.read_csv(os.path.join(folder_path, file), encoding='latin-1', header=None)
print(second_df.head())

"""Drop the index and no_query columns."""

second_df = second_df.drop(columns=3)
second_df = second_df.drop(columns=0)
second_df.rename(columns={1: 'tweet_id', 2: 'tweet_date', 4: 'username', 5: 'tweet_text'}, inplace=True)

"""Concatenate into one dataframe."""

full_df = pd.concat([first_df, second_df], ignore_index=True)

"""Load information on the dataframe."""

print(full_df.info())
print(full_df.head())
print(full_df.tail())

"""Format tweet_id as a string, fill in missing cells with empty strings, and drop any rows with empty tweet_text fields."""

full_df['tweet_id'] = full_df['tweet_id'].astype(str)

full_df['topic'] = full_df['topic'].fillna('')
full_df['tweet_url'] = full_df['tweet_url'].fillna('')
full_df['tweet_date'] = full_df['tweet_date'].fillna('')
full_df['username'] = full_df['username'].fillna('')

full_df = full_df[full_df['tweet_text'] != '']

print(full_df.head())
print(full_df.info())

"""Clean up tweet text column."""

full_df['tweet_text'] = full_df['tweet_text'].str.lower()
full_df['tweet_text'] = full_df['tweet_text'].replace(r'http\S+|www.\S+', '', regex=True)
full_df['tweet_text'] = full_df['tweet_text'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)

print(full_df.head())

full_df['tweet_char_count'] = full_df['tweet_text'].str.len()
print(full_df.info())

"""Extracted topic keyword to fill in empty topic column values."""

!pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('words')
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

english_words = set(words.words())
stop_words = set(stopwords.words('english'))

english_words = set(words.words())
stop_words = set(stopwords.words('english'))

def extract_topic_keywords(text):
    words_in_text = word_tokenize(text.lower())
    filtered_words = [word for word in words_in_text if word.isalnum() and word not in stop_words and word in english_words]
    if len(filtered_words) >= 2:
        return ' '.join(filtered_words[:2])

# Apply only to rows where 'topic' is empty
empty_topic_mask = full_df['topic'] == ''
full_df.loc[empty_topic_mask, 'topic'] = full_df.loc[empty_topic_mask, 'tweet_text'].apply(extract_topic_keywords)

print(full_df.head())
print(full_df.info())

full_df = full_df.drop_duplicates(subset='tweet_text')
full_df = full_df.drop(columns=['tweet_id', 'tweet_url', 'tweet_date', 'username'])

print(full_df.head())
print(full_df['topic'].value_counts())

"""Extracted topics focus on user's status throughout the day. Topic column used to generate GPT tweets, then dropped."""

print(full_df['topic'].value_counts()[:20])

pd.options.display.float_format = '{:.0f}'.format
print(full_df['tweet_char_count'].describe())

full_df = full_df[full_df['tweet_char_count'] <= 280] # Confirm tweets are within character count
print(full_df.describe(include='object'))

output_path = '/content/Twitter_Human_Data.csv'
full_df.to_csv(output_path, index=False)

"""Load GPT-generated tweets."""

gpt_file = '/content/synthetic_tweets_10k.csv'
gpt_tweet_df = pd.read_csv(gpt_file)
gpt_tweet_df = gpt_tweet_df.drop_duplicates(subset='tweet_text')
gpt_tweet_df = gpt_tweet_df.drop(columns='topic')

print(gpt_tweet_df.describe())

"""Combine dataframes and character count column."""

combined_human_ai_df = pd.concat([full_df, gpt_tweet_df], ignore_index=True)
combined_human_ai_df['character_count'] = combined_human_ai_df['tweet_char_count'].combine_first(combined_human_ai_df['char_count'])
combined_human_ai_df = combined_human_ai_df.drop(columns=['tweet_char_count', 'char_count', 'topic'])

print(combined_human_ai_df.head())
print(combined_human_ai_df.info())

"""Drop nulls and output to file."""

combined_human_ai_df = combined_human_ai_df.dropna()
print(combined_human_ai_df.info())

output_path = '/content/Twitter_Human_AI_Data.csv'
combined_human_ai_df.to_csv(output_path, index=False)
