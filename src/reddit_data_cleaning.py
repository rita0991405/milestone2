#!pip install langdetect
#!pip install jsonlines
# Import packages
import pandas as pd
import numpy as np
import json
import re
import unicodedata
from datasets import load_dataset
import random
from langdetect import detect, LangDetectException
import gc
import jsonlines

# Text clean function
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
NON_STANDARD_CHARS = re.compile(r"[^a-z0-9\s.,!?\'-]")
WHITESPACE = re.compile(r"\s+")

def clean_text(text):
    """Clean and normalize text by removing URLs, special characters, and extra whitespace."""
    if not text:
        return ""

    text = URL_PATTERN.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = NON_STANDARD_CHARS.sub("", text)
    text = WHITESPACE.sub(" ", text).strip()

    return text

# Set output file name
output_file = 'cleaned_reddit_data.jsonl'

# Offset tracking for processing in chunks
start_offset = 800000  # Change this between runs (0, 200000, 400000, etc.)
max_per_run = 1000000

# Load dataset in streaming mode
ds = load_dataset("fddemarco/pushshift-reddit", streaming=True)

# Process and save in batches
batch = []
batch_size = 10000

kept_count = 0
removed_count = 0

for i, item in enumerate(ds['train'].skip(start_offset)):
    if i >= max_per_run:
        break
    # Get title and selftext
    title = item.get('title', '')
    selftext = item.get('selftext', '')

    # Clean both
    cleaned_title = clean_text(title)
    cleaned_selftext = clean_text(selftext)

    # Combine title + selftext
    combined_text = (cleaned_title + " " + cleaned_selftext).strip()

    # Only keep records with non-empty text after cleaning
    if combined_text:
        cleaned_item = {
            'text': combined_text,
            'subreddit': item.get('subreddit', ''),
            'score': item.get('score', 0),
            'num_comments': item.get('num_comments', 0),
        }
        batch.append(cleaned_item)
        kept_count += 1
    else:
        removed_count += 1

    # Write and clear batch
    if len(batch) >= batch_size:
        with jsonlines.open(output_file, mode='a') as writer:
            writer.write_all(batch)
        batch.clear()
        gc.collect()
        print(f"Processed {i+1} | Kept: {kept_count} | Removed: {removed_count}")

# Save remaining records
if batch:
    with jsonlines.open(output_file, mode='a') as writer:
        writer.write_all(batch)

print(f"\n=== Final Results ===")
print(f"Offset: {start_offset} to {start_offset + max_per_run}")
print(f"Total processed: {kept_count + removed_count}")
print(f"Kept: {kept_count} records")
print(f"Removed: {removed_count} records")
print(f"Output saved to: {output_file}")

## Add character count to the dataframe
# Read the JSONL file and create DataFrame
data = []

with open('cleaned_reddit_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line.strip())
        data.append({
            'subreddit': entry.get('subreddit', ''),
            'post_text': entry.get('text', ''),
            'char_count': len(entry.get('text', ''))
        })

df = pd.DataFrame(data)

# Save df
df.to_csv('reddit_posts_cleaned.csv', index=False)
print("\nSaved to reddit_posts_cleaned.csv")

# Load the existing reddit_posts_cleaned.csv
df_cleaned = pd.read_csv('/content/reddit_posts_cleaned.csv')

# Load the gpt_redit_posts_10k.csv
df_gpt = pd.read_csv('/content/gpt_reddit_posts_10k.csv')

# Combine the dataframes horizontally (by rows, since columns are the same)
df_combined = pd.concat([df_cleaned, df_gpt], ignore_index=True)

# Save the combined DataFrame to a new CSV file
df_combined.to_csv('combined_reddit_data.csv', index=False)
print("\nCombined data saved to 'combined_reddit_data.csv'")

## Further cleaning (can be adjusted)
# Remove very short texts
df = df[df['word_count'] >= 5]  # At least 5 words

# Remove duplicates
df = df.drop_duplicates(subset=['text'])

# Save cleaned version
df.to_json('cleaned_reddit_final.jsonl', orient='records', lines=True)
