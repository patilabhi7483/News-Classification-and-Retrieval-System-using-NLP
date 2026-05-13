import pandas as pd
import re
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

print("STARTED WORD ANALYSIS")

# Load dataset
data = pd.read_csv("D:/NLP_Project_News/data/bbc_news.csv")

# Combine text
data["text"] = data["title"].astype(str) + " " + data["description"].astype(str)

# Take sample sentences
sentences = data["text"].head(10)

# -----------------------------
# 🔹 REGEX PATTERNS
# -----------------------------

print("\n--- REGEX OUTPUT ---")

for text in sentences:
    # 1. Extract numbers
    numbers = re.findall(r'\d+', text)
    
    # 2. Extract capitalized words (possible names)
    capitals = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # 3. Extract words ending with 'ing'
    ing_words = re.findall(r'\b\w+ing\b', text)
    
    print("\nText:", text[:80])
    print("Numbers:", numbers)
    print("Capital Words:", capitals[:5])
    print("ING words:", ing_words)

# -----------------------------
# 🔹 POS TAGGING
# -----------------------------

print("\n--- POS TAGGING ---")

for text in sentences[:3]:
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    print("\nSentence:", text[:80])
    print("POS Tags:", pos_tags[:10])