import pandas as pd
import nltk
import math
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("STARTED NAIVE BAYES")

# Load dataset
data = pd.read_csv("D:/NLP_Project_News/data/labeled_news.csv")

# -----------------------------
# 🔹 PREPROCESS FUNCTION
# -----------------------------
def preprocess(text):
    return nltk.word_tokenize(text.lower())

# Tokenize all text
data["tokens"] = data["text"].apply(preprocess)

# -----------------------------
# 🔹 TRAIN NAIVE BAYES
# -----------------------------
class_counts = defaultdict(int)
word_counts = defaultdict(lambda: defaultdict(int))
total_words = defaultdict(int)
vocab = set()

for _, row in data.iterrows():
    label = row["label"]
    class_counts[label] += 1

    for word in row["tokens"]:
        word_counts[label][word] += 1
        total_words[label] += 1
        vocab.add(word)

V = len(vocab)
total_docs = len(data)

print("Classes:", dict(class_counts))
print("Vocabulary size:", V)

# -----------------------------
# 🔹 P(class)
# -----------------------------
class_prob = {}
for c in class_counts:
    class_prob[c] = class_counts[c] / total_docs

# -----------------------------
# 🔹 P(word | class) with smoothing
# -----------------------------
def word_prob(word, c):
    return (word_counts[c][word] + 1) / (total_words[c] + V)

# -----------------------------
# 🔹 PREDICTION FUNCTION
# -----------------------------
def predict(text):
    words = preprocess(text)
    scores = {}

    for c in class_prob:
        score = math.log(class_prob[c])

        for word in words:
            score += math.log(word_prob(word, c))

        scores[c] = score

    return max(scores, key=scores.get)

# -----------------------------
# 🔹 TEST CASES
# -----------------------------
print("\n--- TEST PREDICTIONS ---")

# Test 1 (from dataset)
sample = data["text"].iloc[0]
print("\nTest 1 (Dataset Sample)")
print("Text:", sample[:100])
print("Actual:", data["label"].iloc[0])
print("Predicted:", predict(sample))

# Additional test cases
test_cases = [
    "The team won the match with a great score",
    "The president announced new government policies",
    "New AI technology is transforming the industry",
    "The player scored a goal in the final game"
]

for i, text in enumerate(test_cases, start=2):
    print(f"\nTest {i}")
    print("Text:", text)
    print("Predicted:", predict(text))

# -----------------------------
# 🔹 EVALUATION
# -----------------------------
print("\n--- MODEL EVALUATION ---")

# Take random 100 samples
test_data = data.sample(100, random_state=42)

y_true = []
y_pred = []

for _, row in test_data.iterrows():
    actual = row["label"]
    predicted = predict(row["text"])
    
    y_true.append(actual)
    y_pred.append(predicted)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)
# -----------------------------
# 🔹 SENTIMENT ANALYSIS
# -----------------------------

print("\n--- SENTIMENT ANALYSIS ---")

positive_words = {"good", "great", "excellent", "win", "success", "positive"}
negative_words = {"bad", "poor", "fail", "loss", "negative", "problem"}

def sentiment(text):
    words = preprocess(text)
    
    pos_count = 0
    neg_count = 0
    negate = False

    for w in words:
        if w == "not":
            negate = True
            continue

        if w in positive_words:
            if negate:
                neg_count += 1
            else:
                pos_count += 1

        elif w in negative_words:
            if negate:
                pos_count += 1
            else:
                neg_count += 1

        negate = False

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

# Test sentiment
sent_test = [
    "The team had a great win",
    "The government failed to handle the problem",
    "The news report was neutral"
]

for text in sent_test:
    print("\nText:", text)
    print("Sentiment:", sentiment(text))