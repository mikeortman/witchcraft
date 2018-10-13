# Do a quick demo search.
# Input is the embeddings file, args are words to look for nearest neighbors. For demo purposes
from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.nlp.protos.nlpdatatypes_pb2 import WordEmbedding as WordEmbeddingProto
from witchcraft.nlp.datatypes import WordEmbedding

import numpy as np
from sys import argv

embeddings = {}
with open(argv[1], 'rb') as fin:
    for pb_str in protobufs_from_filestream(fin):
        pb: WordEmbeddingProto = WordEmbeddingProto.FromString(pb_str)
        embedding: WordEmbedding = WordEmbedding.from_protobuf(pb)
        embeddings[embedding.get_word()] = embedding.get_embedding()

print("Done. Loaded " + str(len(embeddings)) + " embeddings.")


def search(search_word: str):
    search_embedding = embeddings[search_word]
    denom_search = np.sqrt(np.sum(np.multiply(search_embedding, search_embedding)))
    calculated_distance = []
    for test_word, test_embedding in embeddings.items():
        num = np.sum(np.multiply(search_embedding, test_embedding))
        denom_test = np.sqrt(np.sum(np.multiply(test_embedding, test_embedding)))
        sim = num / denom_search / denom_test

        calculated_distance += [(test_word, sim)]

    calculated_distance.sort(key=lambda a: -a[1])
    return calculated_distance[:10]


def print_search(search_word: str):
    print(search_word)
    for (word, distance) in search(search_word):
        print("\t" + word + ": " + str(distance))

    print("")
    print("")


for word in argv[2:]:
    print_search(word)