# Do a quick demo search.
# Input is the embeddings file, args are words to look for nearest neighbors. For demo purposes
from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.nlp.protos.nlpdatatypes_pb2 import WordEmbedding as WordEmbeddingProto
from witchcraft.nlp.datatypes import WordEmbedding
from witchcraft.nlp.parse import Parser

from flask import Flask, request, jsonify
import tensorflow as tf
from sys import argv


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

session = tf.Session()

@app.route('/search')
def home():
    source_text = request.args.get('text')

    sentenceSequence = Parser.parse_string_to_sentence_sequence(source_text)

    wordsToLookup = [s.get_word_string_normalized() for s in sentenceSequence.get_word_generator() if s.get_word_string_normalized() in embeddingToId]
    wordsToLookupIdx = [embeddingToId[i] for i in wordsToLookup]
    foundIdx, foundStrength = session.run([topTenIdx, topTenVals], {searchWordIndexes: wordsToLookupIdx})

    foundWordMatches = {}
    for i in range(len(wordsToLookup)):
        word = wordsToLookup[i]
        wordSimilar = [idToEmbedding[x] for x in foundIdx[i]]
        wordSimilarStrength = foundStrength[i]
        wordDict = {
            'word': word,
            'similar': []
        }

        for y in range(len(wordSimilar)):
            wordDict['similar'] += [{
                'word': wordSimilar[y],
                'strength': float(wordSimilarStrength[y])
            }]

        foundWordMatches[word] = wordDict

    responseJson = {
        'source_text': source_text,
        'sentences': []
    }


    for sentence in sentenceSequence.get_sentence_generator():
        print(sentence)
        sentenceJson = {
            'sentence_text': str(sentence),
            'words': []
        }



        words = [s for s in argv[2:] if s in embeddingToId]
        wordsIdx = [embeddingToId[i] for i in words]
        foundIdx = session.run(topTenIdx, {searchWordIndexes: wordsIdx})

        for word in sentence.get_word_generator():
            wordNorm = word.get_word_string_normalized()
            if wordNorm in foundWordMatches:
                sentenceJson['words'] += [foundWordMatches[wordNorm]]
            else:
                sentenceJson['words'] += [{
                    'word': wordNorm
                }]

        responseJson['sentences'] += [sentenceJson]

    return jsonify(responseJson)

app.run(host="0.0.0.0", debug=False)