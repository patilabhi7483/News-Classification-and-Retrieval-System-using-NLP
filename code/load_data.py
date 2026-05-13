import pandas as pd

# Load dataset
data = pd.read_csv("D:/NLP_Project_News/data/bbc_news.csv")

# Combine title + description
data["text"] = data["title"] + " " + data["description"]

# Keep only needed column
data = data[["text"]]

print(data.head())
print("Total rows:", len(data))