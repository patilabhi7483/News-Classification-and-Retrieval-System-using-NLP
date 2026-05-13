import nltk
from naive_bayes import predict
from inverted_index import ranked_search

print("=== NLP PIPELINE DEMO ===")

# Input
text = input("Enter news text: ")

# Step 1: Classification
category = predict(text)
print("\nPredicted Category:", category)

# Step 2: Retrieval
results = ranked_search(text)

print("\nTop Results:")
for doc_id, score in results[:3]:
    print(f"\nDoc {doc_id} | Score: {round(score, 3)}")