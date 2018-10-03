# Naughty
Work in progress. This is the first demo program using the witchcraft toolkit. It's goal is to demonstrate a cross-domain mapping between a word and definition using data extracted from UrbanDictionary.

### Implementation details 
Lots of options I'm playing around with:

- User enters a definition and it outputs words or phrases that it thinks would best fit that definition (MT word to definition)
- User enters a word and it outputs a definition for that word, even if that word does not exist in its dictionary (MT definition to word))
- User enters a definition and it returns a list of possible (cross-domain latent space nearest-neighbor search)

Going to try them all.

### Steps
1. Parse the CSV dump from UrbanDictionary, removing junk definitions (too many downvotes, too short, not enough votes)
2. Tokenize each valid definition and word, exporting that data into a set of files of streaming protobuf messages)
3. Word2Vec model on those definitions and words to build word embeddings for the top k words
4. Train models using word embeddings for the different ideas above