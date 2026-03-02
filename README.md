# SIADS 696 - MADS Milestone 2
# Detection and Analysis of AI-Generated Social Media Content

## Project Overview
Machine learning pipeline to detect AI-generated text across social media platforms (Reddit and Twitter) using stylometric, sentiment, and structural features.

## Team Members
- Sung Jin Bae
- Li-yuan Chen
- Jessica Jones

## Repository Structure
- `report/` - Final project report (PDF)
- `data/` - Sample datasets and data source links
- `scr/` - All Python code for data cleaning, modeling, and analysis
- `results/` - Generated figures, model outputs, and evaluation metrics
- `requirements.txt` - Python package dependencies

## Data Sources

### Full Datasets (Too Large for GitHub)
Due to size constraints, full datasets are hosted externally:

1. **Reddit Human Data**: [Pushshift Reddit Dataset](https://huggingface.co/datasets/fddemarco/pushshift-reddit)
   - Source: Hugging Face
   - Size: ~1M posts
   - We used: 60,000 balanced samples

2. **Twitter Human Data**: [Twitter Dataset](https://github.com/kinit-sk/multisocial/blob/main/dataset/)
   - Source: Macko et al. (2025) - MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection
   - Size: 1.6M (2009 corpus, 85MB CSV)
   - We used: 60,000 balanced samples
   - Note: We focused on the larger 2009 dataset for sufficient volume and diversity

3. **AI-Generated Data**: Generated using GPT 5.2
   - Reddit-style: 30,000 posts
   - Twitter-style: 30,000 posts
   - Prompts designed to mimic platform-specific writing styles

### Sample Data (Included)
See `data/sample_data/` for first 100 records from each dataset to demonstrate structure.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Jupyter Notebook (optional, for exploration)

### Installation
```bash
# Clone repository
git clone https://github.com/[your-username]/milestone2.git
cd milestone2

# Install dependencies
pip install -r requirements.txt

# Download SpaCy language model
python -m spacy download en_core_web_sm
```

### Running the Pipeline

1. **Data Cleaning**
```bash
# Clean Reddit data
python scripts/data_cleaning/reddit_data_cleaning.py

# Clean Twitter data
python scripts/data_cleaning/twitter_data_cleaning.py
```

2. **Exploratory Data Analysis & Feature Engineering**
```bash
# Perform EDA and extract features for supervised learning
python scripts/eda_and_feature_engineering.py
# Outputs: data/complete_feature_dataset.csv with all engineered features
```

3. **Supervised Learning**
```bash
python scripts/supervised_learning.py
```

4. **Unsupervised Learning**
```bash
python scripts/unsupervised_learning.py
```

## Key Results
- **Unsupervised clustering reveals separability**: Combining SBERT embeddings with 40+ stylometric and sentiment features, then applying PCA (2 components) + K-Means (k=2), achieved a Silhouette Score of 0.5289, showing meaningful natural separation between AI and human text without labels. Emotional variance and stylistic complexity were primary differentiators.

- **Random Forest achieves state-of-the-art performance**: Among Logistic Regression, SVM, and Random Forest, the Random Forest model performed best with 96.52% accuracy, F1 = 0.9643 ± 0.0009, and AUC-ROC = 0.9923, demonstrating strong generalization across platforms (Twitter/X and Reddit).

- **Emotional variance is the strongest signal**: Feature importance and ablation analyses revealed that subjectivity variance (13.7%) and polarity variance (10.1%) were the most predictive features. Removing punctuation features caused the largest F1 drop (-4.85%), confirming that emotional consistency and expressive patterns are key indicators of AI-generated social media text.

See full report in `report/Report.pdf`

## Technologies Used
- Python 3.12
- scikit-learn 1.3.0
- SpaCy 3.7
- SentenceTransformers (SBERT – all-MiniLM-L6-v2)
- LatentDirichletAllocation (LDA)
- VADER Sentiment
- TextBlob
- pandas, numpy, matplotlib, seaborn
