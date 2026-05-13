import pandas as pd

print("CREATING LABELLED DATA")

# Load dataset
data = pd.read_csv("D:/NLP_Project_News/data/bbc_news.csv")

# Combine text
data["text"] = data["title"].astype(str) + " " + data["description"].astype(str)

# Simple keyword-based labeling
def assign_label(text):
    text = text.lower()

    if any(word in text for word in ["match", "team", "score", "game", "player"]):
        return "sports"
    elif any(word in text for word in ["election", "government", "president", "minister"]):
        return "politics"
    else:
        return "technology"

# Apply labels
data["label"] = data["text"].apply(assign_label)

# Show counts
print(data["label"].value_counts())

# Save new dataset
data.to_csv("D:/NLP_Project_News/data/labeled_news.csv", index=False)

print("Saved as labeled_news.csv")