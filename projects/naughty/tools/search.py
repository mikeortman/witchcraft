# Do a quick demo search.
# Input is the embeddings file, args are words to look for nearest neighbors. For demo purposes
from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.ml.datatypes import PhraseEmbedding, PhraseEmbeddingNgram
from witchcraft.ml.protos.mldatatypes_pb2 import PhraseEmbedding as PhraseEmbeddingProto
from witchcraft.ml.models.nearest_neighbor import NearestNeighborModel, NearestNeighborResult
from witchcraft.nlp.parse import parse_string_to_document

from typing import List
from flask import Flask, request, jsonify
import tensorflow as tf
from sys import argv


app = Flask(__name__)

embeddings = []
phrase_to_embedding = {}
with open(argv[1], 'rb') as fin:
    for pb_str in protobufs_from_filestream(fin):
        pb: PhraseEmbeddingProto = PhraseEmbeddingProto.FromString(pb_str)
        embedding: PhraseEmbedding = PhraseEmbedding.from_protobuf(pb)
        embeddings += [embedding]
        phrase_to_embedding[embedding.get_phrase()] = embedding

search_model: NearestNeighborModel = NearestNeighborModel(embeddings)

print("Done loading embeddings.")

@app.route('/search')
def home():
    source_text = request.args.get('text')

    input_document = parse_string_to_document(source_text)
    document_json = []
    for sentence in input_document:
        sentence_json = {
            'sentence': str(sentence),
            'phrases': []
        }

        for phrase in sentence:
            phrase_norm: str = phrase.to_phrase_normalized()
            nearest: List[NearestNeighborResult] = []
            ngrams: List[PhraseEmbeddingNgram] = []
            if phrase_norm in phrase_to_embedding:
                nearest = search_model.lookup_nearby(phrase_to_embedding[phrase_norm], 10)
                ngrams = phrase_to_embedding[phrase_norm].get_ngrams()

            ngrams_dict = [{
                'ngram': n.get_ngram(),
                'attention': n.get_attention()
            } for n in ngrams]

            sentence_json['phrases'] += [{
                'phrase': str(phrase),
                'norm': phrase_norm,
                'ngrams': ngrams_dict,
                'similar:': [{
                    'phrase': s.get_embedding().get_phrase(),
                    'strength': s.get_strength(),
                    'count': s.get_embedding().get_count()
                } for s in nearest]
            }]

        document_json += [sentence_json]

    return jsonify(document_json)

app.run(host="0.0.0.0", debug=False)