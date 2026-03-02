#!pip install matplotlib-venn pandas numpy matplotlib seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Combine 3 AI-generated text datasets for both reddit and twitter

def combine_datasets(file_paths):
    """Read and combine multiple CSV files into a single dataframe."""
    dataframes = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"Loaded {file_path}: {len(df)} rows")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows in combined dataframe: {len(combined_df)}")
    return combined_df

path = "/content/"

# Combine AI-generated datasets
reddit_files = ['gpt_reddit_posts_10k_1.csv', 'gpt_reddit_posts_10k_2.csv', 'gpt_reddit_posts_10k_3.csv']
twitter_files = ['gpt_tweets_10k_1.csv','gpt_tweets_10k_2.csv','gpt_tweets_10k_3.csv']
ai_reddit = combine_datasets([path + file for file in reddit_files])
ai_twitter = combine_datasets([path + file for file in twitter_files])

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load reddit human data
human_reddit = pd.read_csv('/content/reddit_posts_human.csv')
human_reddit['label'] = 'Human'
human_reddit['source'] = 'Human'
human_reddit['platform'] = 'Reddit'

# Load reddit AI data
ai_reddit = ai_reddit.rename(columns={'generated_content': 'text'})
ai_reddit['label'] = 'AI'
ai_reddit['source'] = 'AI'
ai_reddit['platform'] = 'Reddit'

# Twitter human dataset
human_twitter = pd.read_csv('/content/Twitter_Human_Data.csv')
human_twitter['label'] = 'Human'
human_twitter['source'] = 'Human'
human_twitter['platform'] = 'Twitter'

# Load twitter AI data
ai_twitter = ai_twitter.rename(columns={'generated_content': 'text'})
ai_twitter['label'] = 'AI'
ai_twitter['source'] = 'AI'
ai_twitter['platform'] = 'Twitter'

# Combine
df_combined_reddit = pd.concat([human_reddit, ai_reddit], ignore_index=True)
df_combined_twitter = pd.concat([human_twitter, ai_twitter], ignore_index=True)

## looking at original human data, need downsampling

# BALANCE: Sample equal amounts
n_samples = 30000  # Match AI sample size

human_sampled_reddit = human_reddit.sample(n=n_samples, random_state=42)
ai_sampled_reddit = ai_reddit  # Use all AI data

human_sampled_twitter = human_twitter.sample(n=n_samples, random_state=42)
ai_sampled_twitter = ai_twitter  # Use all AI data

# Combine
df_balanced_reddit = pd.concat([human_sampled_reddit, ai_sampled_reddit], ignore_index=True)
df_balanced_twitter = pd.concat([human_sampled_twitter, ai_sampled_twitter], ignore_index=True)

# Shuffle
df_balanced_reddit = df_balanced_reddit.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced_twitter = df_balanced_twitter.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_balanced_reddit.to_csv('sampled_reddit.csv', index=False)
df_balanced_twitter.to_csv('sampled_twitter.csv', index=False)

## Since twitter dataset has tweet_char_count for human data and char_count for AI, need some cleaning

# For df_balanced_twitter, combine char_count columns
# Combine the two columns into a new one named 'combined_char_count'
df_balanced_twitter['combined_char_count'] = df_balanced_twitter['tweet_char_count'].fillna(df_balanced_twitter['char_count'])

# Drop the original count columns
df_balanced_twitter.drop(columns=['tweet_char_count', 'char_count'], inplace=True)

# Rename the combined column to 'char_count', and tweet_text to 'text'
df_balanced_twitter.rename(columns={'combined_char_count': 'char_count'}, inplace=True)
df_balanced_twitter.rename(columns={'tweet_text': 'text'}, inplace=True)
df_balanced_reddit.rename(columns={'post_text': 'text'}, inplace=True)

## Text Statistics

# Add basic text metrics
df_balanced_reddit['word_count'] = df_balanced_reddit['text'].str.split().str.len()
df_balanced_reddit['sentence_count'] = df_balanced_reddit['text'].str.count(r'[.!?]+') + 1
df_balanced_reddit['avg_word_length'] = df_balanced_reddit['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if x else 0
)

df_balanced_twitter['word_count'] = df_balanced_twitter['text'].str.split().str.len()
df_balanced_twitter['sentence_count'] = df_balanced_twitter['text'].str.count(r'[.!?]+') + 1
df_balanced_twitter['avg_word_length'] = df_balanced_twitter['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if x else 0
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

## So imputing sentence_count didn't really work, try new method

import re
def count_sentences(text):
    if not text or not isinstance(text, str):
        return 1
    # Split on any punctuation followed by whitespace (regardless of capitalization)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    return max(len(sentences), 1)

df_balanced_reddit['word_count'] = df_balanced_reddit['text'].str.split().str.len()
df_balanced_reddit['sentence_count'] = df_balanced_reddit['text'].apply(count_sentences)
df_balanced_reddit['avg_word_length'] = df_balanced_reddit['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if x else 0
)

df_balanced_twitter['word_count'] = df_balanced_twitter['text'].str.split().str.len()
df_balanced_twitter['sentence_count'] = df_balanced_twitter['text'].apply(count_sentences)
df_balanced_twitter['avg_word_length'] = df_balanced_twitter['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if x else 0
)

## conclusion: sentence count still not working due to human data lacking puntuation usages

##Distribution Visualizations
# Reddit visualization
# Character count distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Character count
axes[0, 0].hist(df_balanced_reddit[df_balanced_reddit['label']=='Human']['char_count'],
                bins=50, range=(0, 2000), alpha=0.6, label='Human', color='blue')
axes[0, 0].hist(df_balanced_reddit[df_balanced_reddit['label']=='AI']['char_count'],
                bins=50, range=(0, 2000), alpha=0.6, label='AI', color='red')
axes[0, 0].set_xlabel('Character Count')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Character Count Distribution')
axes[0, 0].legend()

# 2. Word count
axes[0, 1].hist(df_balanced_reddit[df_balanced_reddit['label']=='Human']['word_count'],
                bins=50, range=(0, 400), alpha=0.6, label='Human', color='blue')
axes[0, 1].hist(df_balanced_reddit[df_balanced_reddit['label']=='AI']['word_count'],
                bins=50, range=(0, 400), alpha=0.6, label='AI', color='red')
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Word Count Distribution')
axes[0, 1].legend()

# 3. Sentence count
axes[1, 0].hist(df_balanced_reddit[df_balanced_reddit['label']=='Human']['sentence_count'],
                bins=30, range=(0, 30), alpha=0.6, label='Human', color='blue')
axes[1, 0].hist(df_balanced_reddit[df_balanced_reddit['label']=='AI']['sentence_count'],
                bins=30, range=(0, 30), alpha=0.6, label='AI', color='red')
axes[1, 0].set_xlabel('Sentence Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Sentence Count Distribution')
axes[1, 0].legend()

# 4. Average word length
axes[1, 1].hist(df_balanced_reddit[df_balanced_reddit['label']=='Human']['avg_word_length'],
                bins=30, range=(0, 30), alpha=0.6, label='Human', color='blue')
axes[1, 1].hist(df_balanced_reddit[df_balanced_reddit['label']=='AI']['avg_word_length'],
                bins=30, range=(0, 30), alpha=0.6, label='AI', color='red')
axes[1, 1].set_xlabel('Average Word Length')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Average Word Length Distribution')
axes[1, 1].legend()

plt.suptitle("Reddit Text Statistics")
plt.tight_layout()
plt.show()

# Twitter visualization
# Character count distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Character count
axes[0, 0].hist(df_balanced_twitter[df_balanced_twitter['label']=='Human']['char_count'],
                bins=50, alpha=0.6, label='Human', color='blue')
axes[0, 0].hist(df_balanced_twitter[df_balanced_twitter['label']=='AI']['char_count'],
                bins=50, alpha=0.6, label='AI', color='red')
axes[0, 0].set_xlabel('Character Count')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Character Count Distribution')
axes[0, 0].legend()

# 2. Word count
axes[0, 1].hist(df_balanced_twitter[df_balanced_twitter['label']=='Human']['word_count'],
                bins=50, alpha=0.6, label='Human', color='blue')
axes[0, 1].hist(df_balanced_twitter[df_balanced_twitter['label']=='AI']['word_count'],
                bins=50, alpha=0.6, label='AI', color='red')
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Word Count Distribution')
axes[0, 1].legend()

# 3. Sentence count
axes[1, 0].hist(df_balanced_twitter[df_balanced_twitter['label']=='Human']['sentence_count'],
                bins=30, alpha=0.6, label='Human', color='blue')
axes[1, 0].hist(df_balanced_twitter[df_balanced_twitter['label']=='AI']['sentence_count'],
                bins=30, alpha=0.6, label='AI', color='red')
axes[1, 0].set_xlabel('Sentence Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Sentence Count Distribution')
axes[1, 0].legend()

# 4. Average word length
axes[1, 1].hist(df_balanced_twitter[df_balanced_twitter['label']=='Human']['avg_word_length'],
                bins=30, alpha=0.6, label='Human', color='blue')
axes[1, 1].hist(df_balanced_twitter[df_balanced_twitter['label']=='AI']['avg_word_length'],
                bins=30, alpha=0.6, label='AI', color='red')
axes[1, 1].set_xlabel('Average Word Length')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Average Word Length Distribution')
axes[1, 1].legend()

plt.suptitle("Twitter Text Statistics")
plt.tight_layout()
plt.show()

## Punctuation Analysis

# Function to perform punctuation analysis
def analyze_punctuation(df, dataset_name):
    # Punctuation patterns
    df['exclamation_count'] = df['text'].str.count('!')
    df['question_count'] = df['text'].str.count(r'\?')
    df['comma_count'] = df['text'].str.count(',')
    df['period_count'] = df['text'].str.count(r'\.')
    df['ellipsis_count'] = df['text'].str.count(r'\.\.\.')

    # Normalize by text length
    df['exclamation_ratio'] = df['exclamation_count'] / df['char_count']
    df['question_ratio'] = df['question_count'] / df['char_count']
    df['comma_ratio'] = df['comma_count'] / df['char_count']

    print(f"\n=== {dataset_name} Punctuation Statistics ===")
    print(df.groupby('label')[['exclamation_count', 'question_count',
                                'comma_count', 'period_count']].mean())

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{dataset_name} Punctuation Analysis', fontsize=16, y=1.02)

    df.groupby('label')['exclamation_ratio'].mean().plot(kind='bar', ax=axes[0], color=['red', 'blue'])
    axes[0].set_title('Average Exclamation Ratio')
    axes[0].set_ylabel('Ratio')
    axes[0].set_xlabel('Label')

    df.groupby('label')['question_ratio'].mean().plot(kind='bar', ax=axes[1], color=['red', 'blue'])
    axes[1].set_title('Average Question Ratio')
    axes[1].set_ylabel('Ratio')
    axes[1].set_xlabel('Label')

    df.groupby('label')['comma_ratio'].mean().plot(kind='bar', ax=axes[2], color=['red', 'blue'])
    axes[2].set_title('Average Comma Ratio')
    axes[2].set_ylabel('Ratio')
    axes[2].set_xlabel('Label')

    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_punctuation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df

# Apply to both datasets
df_balanced_reddit = analyze_punctuation(df_balanced_reddit.copy(), 'Reddit')
df_balanced_twitter = analyze_punctuation(df_balanced_twitter.copy(), 'Twitter')

## Sentiment Analysis

#!pip install vaderSentiment

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

tqdm.pandas()

# Function to perform sentiment analysis
def analyze_sentiment(df, dataset_name):
    vader = SentimentIntensityAnalyzer()

    # Basic TextBlob sentiment
    df['polarity'] = df['text'].progress_apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    df['subjectivity'] = df['text'].progress_apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )

    # VADER sentiment
    vader_scores = df['text'].progress_apply(
        lambda x: vader.polarity_scores(x)
    )
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])

    # EMOTIONAL ENGINEERING FEATURES

    def compute_sentiment_variance(text):
        """Calculate sentiment variance across sentences"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > 1:
            sentence_polarities = [TextBlob(s).sentiment.polarity for s in sentences]
            sentence_subjectivities = [TextBlob(s).sentiment.subjectivity for s in sentences]

            return {
                'polarity_variance': np.var(sentence_polarities),
                'polarity_std': np.std(sentence_polarities),
                'subjectivity_variance': np.var(sentence_subjectivities),
                'subjectivity_std': np.std(sentence_subjectivities)
            }
        else:
            return {
                'polarity_variance': 0,
                'polarity_std': 0,
                'subjectivity_variance': 0,
                'subjectivity_std': 0
            }

    variance_features = df['text'].progress_apply(compute_sentiment_variance)
    variance_df = pd.DataFrame(variance_features.tolist())
    df = pd.concat([df, variance_df], axis=1)

    # Additional features
    df['abs_polarity'] = df['polarity'].abs()
    df['is_positive'] = (df['polarity'] > 0.1).astype(int)
    df['is_negative'] = (df['polarity'] < -0.1).astype(int)
    df['is_neutral'] = (df['polarity'].abs() <= 0.1).astype(int)

    # Emotional engineering score (high subjectivity + low variance)
    df['emotional_engineering_score'] = df['subjectivity'] / (df['subjectivity_variance'] + 0.01)

    # STATISTICS
    sentiment_features = [
        'polarity', 'subjectivity', 'vader_compound',
        'polarity_variance', 'subjectivity_variance', 'abs_polarity'
    ]
    print(df.groupby('label')[sentiment_features].describe())

    # Statistical tests
    for feature in sentiment_features:
        human_data = df[df['label']=='Human'][feature].dropna()
        ai_data = df[df['label']=='AI'][feature].dropna()

        t_stat, p_value = ttest_ind(human_data, ai_data)

        significance = ''
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'

    # VISUALIZATIONS
    # Figure 1: Basic sentiment distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{dataset_name} - Basic Sentiment Analysis', fontsize=16, y=1.02)

    df.boxplot(column='polarity', by='label', ax=axes[0])
    axes[0].set_title('Polarity Distribution')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Polarity')

    df.boxplot(column='subjectivity', by='label', ax=axes[1])
    axes[1].set_title('Subjectivity Distribution')
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Subjectivity')

    df.boxplot(column='vader_compound', by='label', ax=axes[2])
    axes[2].set_title('VADER Compound Score')
    axes[2].set_xlabel('Label')
    axes[2].set_ylabel('VADER Compound')

    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_basic_sentiment.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Emotional engineering features
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{dataset_name} - Emotional Engineering Detection', fontsize=16, y=1.00)
    axes = axes.flatten()

    features_to_plot = [
        'polarity_variance', 'subjectivity_variance',
        'abs_polarity', 'polarity_std', 'subjectivity_std',
        'emotional_engineering_score'
    ]

    for i, feature in enumerate(features_to_plot):
        df.boxplot(column=feature, by='label', ax=axes[i])
        axes[i].set_title(feature.replace('_', ' ').title())
        axes[i].set_xlabel('Label')
        axes[i].set_ylabel('Score')

    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_emotional_engineering.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 3: Sentiment distribution (pie charts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{dataset_name} - Sentiment Distribution', fontsize=16, y=1.02)

    # Human sentiment distribution
    human_sentiment = df[df['label']=='Human'][['is_positive', 'is_negative', 'is_neutral']].sum()
    axes[0].pie(human_sentiment, labels=['Positive', 'Negative', 'Neutral'],
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[0].set_title('Human Sentiment Distribution')

    # AI sentiment distribution
    ai_sentiment = df[df['label']=='AI'][['is_positive', 'is_negative', 'is_neutral']].sum()
    axes[1].pie(ai_sentiment, labels=['Positive', 'Negative', 'Neutral'],
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[1].set_title('AI Sentiment Distribution')

    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # EMOTIONAL ENGINEERING INSIGHTS
    print(f"{dataset_name} - Emotional Engineering Analysis")

    human_pol_var = df[df['label']=='Human']['polarity_variance'].mean()
    ai_pol_var = df[df['label']=='AI']['polarity_variance'].mean()

    human_subj_var = df[df['label']=='Human']['subjectivity_variance'].mean()
    ai_subj_var = df[df['label']=='AI']['subjectivity_variance'].mean()

    print(f"\n1. Sentiment Consistency (Lower = More Engineered):")
    print(f"   Polarity Variance:")
    print(f"     Human: {human_pol_var:.6f}")
    print(f"     AI:    {ai_pol_var:.6f}")
    print(f"     → AI is {abs(human_pol_var - ai_pol_var):.6f} {'MORE' if ai_pol_var < human_pol_var else 'LESS'} consistent")

    print(f"\n   Subjectivity Variance:")
    print(f"     Human: {human_subj_var:.6f}")
    print(f"     AI:    {ai_subj_var:.6f}")
    print(f"     → AI is {abs(human_subj_var - ai_subj_var):.6f} {'MORE' if ai_subj_var < human_subj_var else 'LESS'} consistent")

    human_subj = df[df['label']=='Human']['subjectivity'].mean()
    ai_subj = df[df['label']=='AI']['subjectivity'].mean()

    print(f"\n2. Emotional Intensity:")
    print(f"   Subjectivity Score:")
    print(f"     Human: {human_subj:.4f}")
    print(f"     AI:    {ai_subj:.4f}")
    print(f"     → AI is {abs(human_subj - ai_subj):.4f} {'MORE' if ai_subj > human_subj else 'LESS'} emotional")

    print(f"\n3. Sentiment Polarity:")
    print(f"   Positive texts: Human={df[df['label']=='Human']['is_positive'].mean()*100:.1f}%, AI={df[df['label']=='AI']['is_positive'].mean()*100:.1f}%")
    print(f"   Negative texts: Human={df[df['label']=='Human']['is_negative'].mean()*100:.1f}%, AI={df[df['label']=='AI']['is_negative'].mean()*100:.1f}%")
    print(f"   Neutral texts:  Human={df[df['label']=='Human']['is_neutral'].mean()*100:.1f}%, AI={df[df['label']=='AI']['is_neutral'].mean()*100:.1f}%")

    return df

# Apply to both datasets
df_reddit_sentiment = analyze_sentiment(df_balanced_reddit.copy(), 'Reddit')
df_twitter_sentiment = analyze_sentiment(df_balanced_twitter.copy(), 'Twitter')

# Save results
df_reddit_sentiment.to_csv('reddit_with_sentiment.csv', index=False)
df_twitter_sentiment.to_csv('twitter_with_sentiment.csv', index=False)

# Cross-platform Analysis
# Combine datasets
df_reddit_sentiment['platform'] = 'Reddit'
df_twitter_sentiment['platform'] = 'Twitter'

df_combined_sentiment = pd.concat([
    df_reddit_sentiment,
    df_twitter_sentiment
], ignore_index=True)

# Cross-platform visualizations
sentiment_features = [
    'polarity', 'subjectivity', 'vader_compound',
    'polarity_variance', 'subjectivity_variance', 'emotional_engineering_score'
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Cross-Platform Sentiment Comparison', fontsize=16, y=1.00)
axes = axes.flatten()

for i, feature in enumerate(sentiment_features):
    sns.boxplot(data=df_combined_sentiment, x='platform', y=feature,
                hue='label', ax=axes[i])
    axes[i].set_title(feature.replace('_', ' ').title())
    axes[i].set_ylabel('Score')
    axes[i].set_xlabel('Platform')

plt.tight_layout()
plt.show()

# Summary table
summary = df_combined_sentiment.groupby(['platform', 'label'])[sentiment_features].mean()

# Save combined dataset
df_combined_sentiment.to_csv('combined_with_sentiment_features.csv', index=False)

## Structural Metrics (Burstiness)

def calculate_burstiness(text):
    """Calculate sentence length variance (burstiness)"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return 0

    sent_lengths = [len(s.split()) for s in sentences]
    return np.var(sent_lengths)

df_balanced_reddit['burstiness'] = df_balanced_reddit['text'].progress_apply(calculate_burstiness)
df_balanced_twitter['burstiness'] = df_balanced_twitter['text'].progress_apply(calculate_burstiness)

print("\n=== Burstiness (Sentence Length Variance) of Reddit ===")
print(df_balanced_reddit.groupby('label')['burstiness'].describe())

print("\n=== Burstiness (Sentence Length Variance) of Twitter ===")
print(df_balanced_twitter.groupby('label')['burstiness'].describe())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Burstiness Analysis: Sentence Length Variance', fontsize=16)

# Reddit Plot (Left)
df_balanced_reddit.boxplot(column='burstiness', by='label', ax=axes[0])
axes[0].set_title('Reddit')
axes[0].set_ylabel('Variance')

# Twitter Plot (Right)
df_balanced_twitter.boxplot(column='burstiness', by='label', ax=axes[1])
axes[1].set_title('Twitter')
axes[1].set_ylabel('Variance')

# Clean up the layout
plt.suptitle('Burstiness (Sentence Length Variance)', fontsize=16) # Main title
fig.get_axes()[0].set_xlabel('') # Optional: removes redundant 'label' text
fig.get_axes()[1].set_xlabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust to make room for suptitle
plt.show()

## Since human data is lacking punctation marks, 
# it's true that sentence burstiness is not going to work. 
# We can re-define the burstiness by "chunking" using spaCy
import spacy

# Load the small English model
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def calculate_burstiness_spacy(text):
    if not isinstance(text, str) or not text.strip():
        return 0

    # Process the text with spaCy
    doc = nlp(text)

    # Extract sentence lengths (in words)
    # spaCy's .sents is a generator of sentence spans
    sent_lengths = [len(sent) for sent in doc.sents]

    if len(sent_lengths) <= 1:
        return 0

    # Return variance (or np.std(sent_lengths) / np.mean(sent_lengths) for CV)
    return np.var(sent_lengths)

# Apply to your dataframe
df_balanced_reddit['burstiness'] = df_balanced_reddit['text'].progress_apply(calculate_burstiness_spacy)
df_balanced_twitter['burstiness'] = df_balanced_twitter['text'].progress_apply(calculate_burstiness_spacy)

# spacy Visualize
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Burstiness Analysis: Sentence Length Variance', fontsize=16)

# Reddit Plot (Left)
df_balanced_reddit.boxplot(column='burstiness', by='label', ax=axes[0])
axes[0].set_title('Reddit')
axes[0].set_ylabel('Variance')

# Twitter Plot (Right)
df_balanced_twitter.boxplot(column='burstiness', by='label', ax=axes[1])
axes[1].set_title('Twitter')
axes[1].set_ylabel('Variance')

# Clean up the layout
plt.suptitle('Burstiness (Sentence Length Variance)', fontsize=16)
fig.get_axes()[0].set_xlabel('')
fig.get_axes()[1].set_xlabel('')

plt.show()

## Syntactic Complexity: POS tags (via SpaCy) to calculate the ratio of descriptive adjectives to functional nouns"""

def extract_pos_features(text):
    """
    Extract POS tag features including adjective-to-noun ratio
    """
    doc = nlp(text)

    # Count POS tags
    pos_counts = {
        'ADJ': 0,    # Adjectives (descriptive)
        'NOUN': 0,   # Nouns (functional)
        'VERB': 0,   # Verbs
        'ADV': 0,    # Adverbs
        'PRON': 0,   # Pronouns
        'DET': 0,    # Determiners (the, a, an)
        'ADP': 0,    # Adpositions (prepositions)
        'CONJ': 0,   # Conjunctions
        'NUM': 0,    # Numbers
        'PUNCT': 0,  # Punctuation
    }

    # Count each POS tag
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1

    # Total word count (excluding punctuation)
    total_words = len([token for token in doc if not token.is_punct and not token.is_space])

    # Calculate ratios
    features = {
        'adj_count': pos_counts['ADJ'],
        'noun_count': pos_counts['NOUN'],
        'verb_count': pos_counts['VERB'],
        'adv_count': pos_counts['ADV'],
        'total_words': total_words,

        # Key metric: Adjective-to-Noun Ratio
        'adj_noun_ratio': pos_counts['ADJ'] / pos_counts['NOUN'] if pos_counts['NOUN'] > 0 else 0,

        # Additional useful ratios
        'adj_ratio': pos_counts['ADJ'] / total_words if total_words > 0 else 0,
        'noun_ratio': pos_counts['NOUN'] / total_words if total_words > 0 else 0,
        'verb_ratio': pos_counts['VERB'] / total_words if total_words > 0 else 0,
        'adv_ratio': pos_counts['ADV'] / total_words if total_words > 0 else 0,
    }

    return features

# Apply to all texts with progress bar
tqdm.pandas()
reddit_pos_features = df_balanced_reddit['text'].progress_apply(extract_pos_features)
twitter_pos_features = df_balanced_twitter['text'].progress_apply(extract_pos_features)

# Convert to DataFrame
reddit_pos_df = pd.DataFrame(reddit_pos_features.tolist())
twitter_pos_df = pd.DataFrame(twitter_pos_features.tolist())

# Add to original dataframe
reddit_df_with_pos = pd.concat([df_balanced_reddit, reddit_pos_df], axis=1)
twitter_df_with_pos = pd.concat([df_balanced_twitter, twitter_pos_df], axis=1)
# Save
reddit_df_with_pos.to_csv('reddit_pos_features.csv', index=False)
twitter_df_with_pos.to_csv('twitter_pos_features.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

print("\n" + "="*80)
print("REDDIT DATASET - Adjective-to-Noun Ratio Analysis")
print("="*80)

# Statistics
print("\nStatistics by Label (Reddit):")
print(reddit_df_with_pos.groupby('label')['adj_noun_ratio'].describe())

# Statistical test
reddit_human = reddit_df_with_pos[reddit_df_with_pos['label']=='Human']['adj_noun_ratio']
reddit_ai = reddit_df_with_pos[reddit_df_with_pos['label']=='AI']['adj_noun_ratio']

t_stat, p_value = ttest_ind(reddit_human, reddit_ai)
print(f"\nT-test results (Reddit):")
print(f"  Human mean: {reddit_human.mean():.4f}")
print(f"  AI mean: {reddit_ai.mean():.4f}")
print(f"  Difference: {abs(reddit_human.mean() - reddit_ai.mean()):.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'Yes ***' if p_value < 0.001 else 'Yes **' if p_value < 0.01 else 'Yes *' if p_value < 0.05 else 'No'}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot
reddit_df_with_pos.boxplot(column='adj_noun_ratio', by='label', ax=axes[0])
axes[0].set_title('Reddit: Adjective-to-Noun Ratio')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Adj/Noun Ratio')

# Histogram
axes[1].hist(reddit_human, bins=50, alpha=0.6, label='Human', color='blue', edgecolor='black')
axes[1].hist(reddit_ai, bins=50, alpha=0.6, label='AI', color='red', edgecolor='black')
axes[1].set_xlabel('Adjective-to-Noun Ratio')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Reddit: Distribution')
axes[1].legend()

plt.suptitle('')
plt.tight_layout()
plt.savefig('reddit_adj_noun_ratio.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("TWITTER DATASET - Adjective-to-Noun Ratio Analysis")
print("="*80)

# Statistics
print("\nStatistics by Label (Twitter):")
print(twitter_df_with_pos.groupby('label')['adj_noun_ratio'].describe())

# Statistical test
twitter_human = twitter_df_with_pos[twitter_df_with_pos['label']=='Human']['adj_noun_ratio']
twitter_ai = twitter_df_with_pos[twitter_df_with_pos['label']=='AI']['adj_noun_ratio']

t_stat, p_value = ttest_ind(twitter_human, twitter_ai)
print(f"\nT-test results (Twitter):")
print(f"  Human mean: {twitter_human.mean():.4f}")
print(f"  AI mean: {twitter_ai.mean():.4f}")
print(f"  Difference: {abs(twitter_human.mean() - twitter_ai.mean()):.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'Yes ***' if p_value < 0.001 else 'Yes **' if p_value < 0.01 else 'Yes *' if p_value < 0.05 else 'No'}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot
twitter_df_with_pos.boxplot(column='adj_noun_ratio', by='label', ax=axes[0])
axes[0].set_title('Twitter: Adjective-to-Noun Ratio')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Adj/Noun Ratio')

# Histogram
axes[1].hist(twitter_human, bins=50, alpha=0.6, label='Human', color='blue', edgecolor='black')
axes[1].hist(twitter_ai, bins=50, alpha=0.6, label='AI', color='red', edgecolor='black')
axes[1].set_xlabel('Adjective-to-Noun Ratio')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Twitter: Distribution')
axes[1].legend()

plt.suptitle('')
plt.tight_layout()
plt.savefig('twitter_adj_noun_ratio.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("CROSS-PLATFORM COMPARISON")
print("="*80)

# Summary table
comparison_data = {
    'Platform': ['Reddit', 'Reddit', 'Twitter', 'Twitter'],
    'Label': ['Human', 'AI', 'Human', 'AI'],
    'Mean Adj/Noun Ratio': [
        reddit_human.mean(),
        reddit_ai.mean(),
        twitter_human.mean(),
        twitter_ai.mean()
    ],
    'Std Adj/Noun Ratio': [
        reddit_human.std(),
        reddit_ai.std(),
        twitter_human.std(),
        twitter_ai.std()
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Combined box plot
data_for_plot = pd.concat([
    reddit_df_with_pos.assign(platform='Reddit'),
    twitter_df_with_pos.assign(platform='Twitter')
])

sns.boxplot(data=data_for_plot, x='platform', y='adj_noun_ratio', hue='label', ax=axes[0])
axes[0].set_title('Adjective-to-Noun Ratio by Platform and Label')
axes[0].set_ylabel('Adj/Noun Ratio')
axes[0].set_xlabel('Platform')

# Bar chart of means
comparison_df.pivot(index='Platform', columns='Label', values='Mean Adj/Noun Ratio').plot(
    kind='bar', ax=axes[1], color=['blue', 'red'], edgecolor='black'
)
axes[1].set_title('Mean Adjective-to-Noun Ratio Comparison')
axes[1].set_ylabel('Mean Adj/Noun Ratio')
axes[1].set_xlabel('Platform')
axes[1].legend(title='Label')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('platform_comparison_adj_noun.png', dpi=300, bbox_inches='tight')
plt.show()

# Combine both platforms for supervised learning
df_combined = pd.concat([
    reddit_df_with_pos.assign(platform='Reddit'),
    twitter_df_with_pos.assign(platform='Twitter')
], ignore_index=True)

# Shuffle
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
df_combined.to_csv('combined_with_pos_features.csv', index=False)

print(f"\n Combined dataset created: {len(df_combined)} samples")
print(f"\nDistribution:")
print(df_combined.groupby(['platform', 'label']).size())

# Overall statistics
print("\n" + "="*80)
print("OVERALL STATISTICS (Combined)")
print("="*80)
print(df_combined.groupby('label')['adj_noun_ratio'].describe())

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# 1. Is the pattern consistent across platforms?
print("\n1. Pattern Consistency:")
reddit_diff = reddit_ai.mean() - reddit_human.mean()
twitter_diff = twitter_ai.mean() - twitter_human.mean()

print(f"   Reddit: AI {'higher' if reddit_diff > 0 else 'lower'} by {abs(reddit_diff):.4f}")
print(f"   Twitter: AI {'higher' if twitter_diff > 0 else 'lower'} by {abs(twitter_diff):.4f}")
print(f"   Pattern consistent: {'Yes' if (reddit_diff > 0) == (twitter_diff > 0) else 'No'}")

# 2. Which platform shows stronger separation?
print(f"\n2. Stronger Separation:")
print(f"   Reddit difference: {abs(reddit_diff):.4f}")
print(f"   Twitter difference: {abs(twitter_diff):.4f}")
print(f"   Stronger in: {'Reddit' if abs(reddit_diff) > abs(twitter_diff) else 'Twitter'}")

# 3. Platform baseline differences
print(f"\n3. Platform Baseline (Human texts):")
print(f"   Reddit Human: {reddit_human.mean():.4f}")
print(f"   Twitter Human: {twitter_human.mean():.4f}")
print(f"   Difference: {abs(reddit_human.mean() - twitter_human.mean()):.4f}")
print(f"   (This shows natural platform writing style differences)")

## Combining All Features into complete_feature_dataset.csv"""
# Load all feature-extracted datasets
# Reddit with POS features
df_reddit_pos = pd.read_csv('/content/reddit_pos_features.csv')
# Twitter with POS features
df_twitter_pos = pd.read_csv('/content/twitter_pos_features.csv')
# Reddit with sentiment features
df_reddit_sentiment = pd.read_csv('/content/reddit_with_sentiment.csv')
# Twitter with sentiment features
df_twitter_sentiment = pd.read_csv('/content/twitter_with_sentiment.csv')

# Merge Reddit datasets
# Assuming both have 'text' column and are in the same order
# Option A: If they're already aligned (same order)
if len(df_reddit_pos) == len(df_reddit_sentiment) and (df_reddit_pos['text'] == df_reddit_sentiment['text']).all():
    # Get unique columns from sentiment (exclude duplicates)
    sentiment_cols = [col for col in df_reddit_sentiment.columns
                     if col not in df_reddit_pos.columns or col == 'text']

    df_reddit_combined = df_reddit_pos.copy()
    for col in sentiment_cols:
        if col not in df_reddit_combined.columns:
            df_reddit_combined[col] = df_reddit_sentiment[col].values
else:
    # Option B: Merge on text column
    df_reddit_combined = df_reddit_pos.merge(
        df_reddit_sentiment,
        on='text',
        how='inner',
        suffixes=('', '_sentiment')
    )
    # Remove duplicate columns with suffix
    df_reddit_combined = df_reddit_combined[[col for col in df_reddit_combined.columns
                                             if not col.endswith('_sentiment')]]

# Merge Twitter datasets
if len(df_twitter_pos) == len(df_twitter_sentiment) and (df_twitter_pos['text'] == df_twitter_sentiment['text']).all():
    sentiment_cols = [col for col in df_twitter_sentiment.columns
                     if col not in df_twitter_pos.columns or col == 'text']

    df_twitter_combined = df_twitter_pos.copy()
    for col in sentiment_cols:
        if col not in df_twitter_combined.columns:
            df_twitter_combined[col] = df_twitter_sentiment[col].values
else:
    df_twitter_combined = df_twitter_pos.merge(
        df_twitter_sentiment,
        on='text',
        how='inner',
        suffixes=('', '_sentiment')
    )
    df_twitter_combined = df_twitter_combined[[col for col in df_twitter_combined.columns
                                               if not col.endswith('_sentiment')]]
# Add platform identifier
df_reddit_combined['platform'] = 'Reddit'
df_twitter_combined['platform'] = 'Twitter'
# Combine both platforms
df_complete = pd.concat([df_reddit_combined, df_twitter_combined], ignore_index=True)
# Add binary label
df_complete['label_binary'] = (df_complete['label'] == 'AI').astype(int)

# Extract all features
def extract_all_features(df):
    """
    Extract comprehensive stylometric, syntactic, sentiment, and structural features
    """
    # BASIC STYLOMETRIC FEATURES
    df['char_count'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['sentence_count'] = df['text'].str.count(r'[.!?]+') + 1
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    df['avg_sentence_length'] = df['word_count'] / df['sentence_count']
    
    # PUNCTUATION PATTERNS
    df['exclamation_count'] = df['text'].str.count('!')
    df['question_count'] = df['text'].str.count(r'\?')
    df['comma_count'] = df['text'].str.count(',')
    df['period_count'] = df['text'].str.count(r'\.')
    df['ellipsis_count'] = df['text'].str.count(r'\.\.\.')
    
    # Normalize by length
    df['exclamation_ratio'] = df['exclamation_count'] / df['char_count']
    df['question_ratio'] = df['question_count'] / df['char_count']
    df['comma_ratio'] = df['comma_count'] / df['char_count']
    df['punctuation_density'] = (
        df['exclamation_count'] + df['question_count'] + 
        df['comma_count'] + df['period_count']
    ) / df['char_count']
    
    # VOCABULARY RICHNESS
    # Type-Token Ratio
    df['ttr'] = df['text'].apply(
        lambda x: len(set(x.lower().split())) / len(x.split()) if x and len(x.split()) > 0 else 0
    )
    
    # STRUCTURAL METRICS (BURSTINESS)
    def calculate_burstiness(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 1:
            return 0, 0
        sent_lengths = [len(s.split()) for s in sentences]
        return np.var(sent_lengths), np.std(sent_lengths)
    
    burstiness_features = df['text'].apply(calculate_burstiness)
    df['sentence_length_variance'] = burstiness_features.apply(lambda x: x[0])
    df['sentence_length_std'] = burstiness_features.apply(lambda x: x[1])
    
    # === TYPO/UNCOMMON WORD PROXY ===
    # Count words with numbers or special chars (rough proxy for typos)
    df['special_word_ratio'] = df['text'].apply(
        lambda x: len(re.findall(r'\b\w*[0-9]\w*\b', x)) / len(x.split()) if x and len(x.split()) > 0 else 0
    )
    
    # PLATFORM-SPECIFIC FEATURES
    df['platform_reddit'] = (df['platform'] == 'Reddit').astype(int)
    df['platform_twitter'] = (df['platform'] == 'Twitter').astype(int)
    
    return df

# Extract features if not already present
if 'ttr' not in df_complete.columns:
    df_complete = extract_all_features(df_complete)

# Save complete dataset
df_complete.to_csv('complete_feature_dataset.csv', index=False)
