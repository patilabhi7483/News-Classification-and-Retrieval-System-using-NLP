import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print("STARTED")

# Load dataset (⚠️ CHANGE FILE NAME if needed)
data = pd.read_csv("D:/NLP_Project_News/data/bbc_news.csv")

# Debug: check data loading
print("\n--- DATA HEAD ---")
print(data.head())
print("Rows:", len(data))

# Combine title + description
data["text"] = data["title"].astype(str) + " " + data["description"].astype(str)

# Load stopwords ONCE (important fix)
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    words = []
    for word in tokens:
        if word not in stop_words and word not in string.punctuation:
            words.append(word)
    
    return words

# Debug: test single row
print("\n--- SINGLE TEXT ---")
print(data["text"].iloc[0])

print("\n--- PROCESSED TEXT ---")
print(preprocess(data["text"].iloc[0]))

# Apply preprocessing (ONLY 5 rows for now → faster)
data["processed"] = data["text"].head(5).apply(preprocess)

# Final output
print("\n--- FINAL OUTPUT ---")
print(data[["text", "processed"]].head())