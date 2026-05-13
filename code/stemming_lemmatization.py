import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

print("STARTED STEMMING vs LEMMATIZATION")

# Initialize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Sample words (you can expand to 20 later for report)
words = [
    "running", "better", "studies", "playing", "went",
    "cars", "flying", "easily", "ate", "children"
]

print("\nWord\t\tStemmed\t\tLemmatized")

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    
    print(f"{word}\t\t{stem}\t\t{lemma}")