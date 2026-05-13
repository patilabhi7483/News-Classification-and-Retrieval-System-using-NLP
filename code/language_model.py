import pandas as pd
import nltk
from collections import defaultdict, Counter
import math
import random

print("STARTED FINAL LANGUAGE MODEL")

# Load dataset
data = pd.read_csv("D:/NLP_Project_News/data/bbc_news.csv")

# Combine text
data["text"] = data["title"].astype(str) + " " + data["description"].astype(str)

# Shuffle data
sentences = data["text"].tolist()
random.shuffle(sentences)

# Split 80% train, 20% test
split = int(0.8 * len(sentences))
train_data = sentences[:split]
test_data = sentences[split:]

print("Train size:", len(train_data))
print("Test size:", len(test_data))

# -----------------------------
# 🔹 TOKENIZE TRAIN DATA
# -----------------------------
tokens = []
for sentence in train_data:
    tokens.extend(nltk.word_tokenize(sentence.lower()))

vocab = set(tokens)
V = len(vocab)

print("Vocabulary size:", V)

# -----------------------------
# 🔹 UNIGRAM + BIGRAM COUNTS
# -----------------------------
unigram_counts = Counter(tokens)
bigram_counts = defaultdict(lambda: defaultdict(int))

for i in range(len(tokens) - 1):
    w1 = tokens[i]
    w2 = tokens[i + 1]
    bigram_counts[w1][w2] += 1

# -----------------------------
# 🔹 BIGRAM PROBABILITY (MLE)
# -----------------------------
def bigram_prob(w1, w2):
    return bigram_counts[w1][w2] / unigram_counts[w1] if unigram_counts[w1] > 0 else 0

# -----------------------------
# 🔹 LAPLACE SMOOTHING
# -----------------------------
def smoothed_bigram_prob(w1, w2):
    return (bigram_counts[w1][w2] + 1) / (unigram_counts[w1] + V)

# -----------------------------
# 🔹 PERPLEXITY FUNCTION
# -----------------------------
def compute_perplexity(dataset, prob_func):
    log_prob = 0
    N = 0

    for sentence in dataset[:50]:  # use 50 sentences for speed
        words = nltk.word_tokenize(sentence.lower())
        N += len(words)

        for i in range(len(words) - 1):
            p = prob_func(words[i], words[i+1])
            if p == 0:
                p = 1e-6
            log_prob += math.log(p)

    return math.exp(-log_prob / N)

# -----------------------------
# 🔹 RESULTS
# -----------------------------
print("\n--- PERPLEXITY WITHOUT SMOOTHING ---")
pp1 = compute_perplexity(test_data, bigram_prob)
print("Perplexity:", pp1)

print("\n--- PERPLEXITY WITH LAPLACE SMOOTHING ---")
pp2 = compute_perplexity(test_data, smoothed_bigram_prob)
print("Perplexity:", pp2)