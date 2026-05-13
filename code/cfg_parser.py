import nltk
from nltk import CFG
from nltk.parse import ChartParser

print("STARTED CFG PARSING")

# Define grammar
grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N | Det Adj N | N
VP -> V NP | V
Det -> 'the' | 'a'
N -> 'president' | 'country' | 'war' | 'people' | 'news'
V -> 'says' | 'leads' | 'affects'
Adj -> 'big' | 'new'
""")

# Create parser
parser = ChartParser(grammar)

# Example sentences
sentences = [
    "the president says",
    "the country affects people"
]

for sentence in sentences:
    print("\nSentence:", sentence)
    words = sentence.split()

    for tree in parser.parse(words):
        print(tree)
        tree.pretty_print()