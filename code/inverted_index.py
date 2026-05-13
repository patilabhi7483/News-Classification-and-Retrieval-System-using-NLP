import pandas as pd
import nltk
from collections import defaultdict

print("STARTED INVERTED INDEX")

# Load dataset
data = pd.read_csv("D:/NLP_Project_News/data/labeled_news.csv")

# Combine text
data["text"] = data["text"].astype(str)

# Preprocess
def preprocess(text):
    return nltk.word_tokenize(text.lower())

# Build inverted index
inverted_index = defaultdict(set)

for idx, row in data.iterrows():
    words = preprocess(row["text"])

    for word in words:
        inverted_index[word].add(idx)

# Convert sets to list
for word in inverted_index:
    inverted_index[word] = list(inverted_index[word])

# Show sample
print("\n--- SAMPLE INDEX ---")
for i, (word, docs) in enumerate(inverted_index.items()):
    print(word, "→", docs[:5])
    if i == 10:
        break

# -----------------------------
# 🔹 BOOLEAN RETRIEVAL
# -----------------------------

print("\n--- BOOLEAN RETRIEVAL ---")

def boolean_search(query):
    tokens = query.lower().split()

    if "and" in tokens:
        w1, w2 = tokens[0], tokens[2]
        return list(set(inverted_index.get(w1, [])) & set(inverted_index.get(w2, [])))

    elif "or" in tokens:
        w1, w2 = tokens[0], tokens[2]
        return list(set(inverted_index.get(w1, [])) | set(inverted_index.get(w2, [])))

    elif "not" in tokens:
        w1, w2 = tokens[0], tokens[2]
        return list(set(inverted_index.get(w1, [])) - set(inverted_index.get(w2, [])))

    else:
        return inverted_index.get(tokens[0], [])

# -----------------------------
# 🔹 TEST QUERIES
# -----------------------------

queries = [
    "war AND ukraine",
    "team OR match",
    "war NOT russia"
]

for q in queries:
    result = boolean_search(q)
    print("\nQuery:", q)
    print("Documents:", result[:5])

# -----------------------------
# 🔹 TF-IDF RANKED RETRIEVAL
# -----------------------------
import math
from nltk.corpus import wordnet

def expand_query(query):
    expanded = set(query.lower().split())

    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))

    return list(expanded)

print("\n--- TF-IDF RETRIEVAL ---")

# Preprocess function (reuse)
def preprocess(text):
    return nltk.word_tokenize(text.lower())

# Total documents
N = len(data)

# Document frequency (df)
df = defaultdict(int)

for word in inverted_index:
    df[word] = len(inverted_index[word])

# TF-IDF calculation
def tf_idf_score(query, doc_id):
    words = preprocess(query)
    score = 0

    doc_words = preprocess(data.iloc[doc_id]["text"])
    doc_len = len(doc_words)

    for word in words:
        tf = doc_words.count(word) / doc_len if doc_len > 0 else 0
        idf = math.log(N / (df[word] + 1)) if word in df else 0
        score += tf * idf

    return score

# Search function
def ranked_search(query):
    scores = []

    for doc_id in range(len(data)):
        score = tf_idf_score(query, doc_id)
        if score > 0:
            scores.append((doc_id, score))

    # sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# -----------------------------
# 🔹 TEST QUERY
# -----------------------------

query = "ukraine war"

expanded = expand_query(query)
results = ranked_search(" ".join(expanded))

print("\nQuery:", query)
for doc_id, score in results[:5]:
    print(f"Doc {doc_id} | Score: {round(score, 4)}")
    print(data.iloc[doc_id]["text"][:80], "\n")

# -----------------------------
# 🔹 WORDNET QUERY EXPANSION
# -----------------------------
from nltk.corpus import wordnet

print("\n--- WORDNET QUERY EXPANSION ---")

def expand_query(query):
    expanded = set(query.lower().split())

    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))

    return list(expanded)

# Test
query = "war"
expanded_query = expand_query(query)

print("Original Query:", query)
print("Expanded Query:", expanded_query[:10])

# -----------------------------
# 🔹 PRECISION@5
# -----------------------------

print("\n--- PRECISION@5 ---")

def precision_at_k(results, relevant_docs, k=5):
    results = [doc_id for doc_id, _ in results[:k]]
    relevant = sum(1 for doc in results if doc in relevant_docs)
    return relevant / k

# Example relevant docs (you define manually)
query = "war"
relevant_docs = [0, 1, 2]  # manually assume relevant docs

results = ranked_search(query)

p5 = precision_at_k(results, relevant_docs, 5)

print("Query:", query)
print("Precision@5:", p5)