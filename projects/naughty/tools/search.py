# Do a quick demo search.
# Input is the embeddings file, args are words to look for nearest neighbors. For demo purposes
from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.nlp.protos.nlpdatatypes_pb2 import WordEmbedding as WordEmbeddingProto
from witchcraft.nlp.datatypes import WordEmbedding

import tensorflow as tf
from sys import argv


from flask import Flask
app = Flask(__name__)

embeddingMatrix = []
embeddingToId = {}
idToEmbedding = []
with open(argv[1], 'rb') as fin:
    i = 0
    for pb_str in protobufs_from_filestream(fin):
        pb: WordEmbeddingProto = WordEmbeddingProto.FromString(pb_str)
        embedding: WordEmbedding = WordEmbedding.from_protobuf(pb)
        embeddingToId[embedding.get_word()] = i
        idToEmbedding += [embedding.get_word()]
        embeddingMatrix += [embedding.get_embedding()]
        i += 1

print("Done. Loaded " + str(len(embeddingToId)) + " embeddings.")

print ("Building graph.")
embeddingsModel = tf.constant(embeddingMatrix, dtype=tf.float32)
embeddingsNormal = tf.sqrt(tf.reduce_sum(tf.square(embeddingsModel), axis=-1))

totalEmbeddings = embeddingsModel.shape[0]
embeddingSize = embeddingsModel.shape[1]
searchWordIndexes = tf.placeholder(tf.int32, shape=[None])
searchEmbeddings = tf.gather(embeddingsModel, searchWordIndexes)
searchEmbeddingsNormal = tf.gather(embeddingsNormal, searchWordIndexes)

searchDotProduct = tf.reduce_sum(tf.expand_dims(searchEmbeddings, axis=1) * embeddingsModel, axis=-1)
cosineSim = searchDotProduct / embeddingsNormal / tf.expand_dims(searchEmbeddingsNormal, axis=-1)
topTenVals, topTenIdx = tf.nn.top_k(cosineSim, k=10, sorted=True)


print("search word indexes: " + str(searchWordIndexes.shape))
print("search embeddings: " + str(searchEmbeddings.shape))
print("search normal: " + str(searchEmbeddingsNormal.shape))
print("embeddings model: " + str(embeddingsModel.shape))
print("embeddings normal: " + str(embeddingsNormal.shape))
print("dot product: " + str(searchDotProduct.shape))
print("cosim: " + str(cosineSim.shape))
print("topten: " + str(topTenVals.shape) + ", " + str(topTenIdx.shape))

session = tf.Session()


@app.route('/')
def home():
    print("running.")
    words = [s for s in argv[2:] if s in embeddingToId]
    wordsIdx = [embeddingToId[i] for i in words]
    foundIdx = session.run(topTenIdx, {searchWordIndexes: wordsIdx})

    for wordIndexes in foundIdx:
        print("Word:")
        print("Found: " + str([idToEmbedding[r] for r in wordIndexes]))
    return 'Done'

app.run(debug=True)

# print(cosineSim.shape)
# print(searchDotProduct.shape)
# print(embeddingsModel.shape)
# print(tf.expand_dims(embeddingsNormal, axis=-1).shape)
# print(searchEmbeddings.shape)
# print(tf.expand_dims(searchEmbeddingsNormal, axis=-1).shape)



# embeddings = {}
# with open(argv[1], 'rb') as fin:
#     for pb_str in protobufs_from_filestream(fin):
#         pb: WordEmbeddingProto = WordEmbeddingProto.FromString(pb_str)
#         embedding: WordEmbedding = WordEmbedding.from_protobuf(pb)
#         embeddings[embedding.get_word()] = embedding.get_embedding()
#
# print("Done. Loaded " + str(len(embeddings)) + " embeddings.")


# def search(search_word: str):
#     search_embedding = embeddings[search_word]
#     denom_search = np.sqrt(np.sum(np.multiply(search_embedding, search_embedding)))
#     calculated_distance = []
#     for test_word, test_embedding in embeddings.items():
#         num = np.sum(np.multiply(search_embedding, test_embedding))
#         denom_test = np.sqrt(np.sum(np.multiply(test_embedding, test_embedding)))
#         sim = num / denom_search / denom_test
#
#         calculated_distance += [(test_word, sim)]
#
#     calculated_distance.sort(key=lambda a: -a[1])
#     return calculated_distance[:10]
#
#
# def print_search(search_word: str):
#     print(search_word)
#     for (word, distance) in search(search_word):
#         print("\t" + word + ": " + str(distance))
#
#     print("")
#     print("")
#
#
# for word in argv[2:]:
#     print_search(word)

