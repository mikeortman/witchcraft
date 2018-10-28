from sys import argv
from typing import Optional, List

from witchcraft.util.protobuf import protobufs_from_filestream
# from witchcraft.ml.models.word2vec import Word2VecVocabBuilder, Word2VecVocab, Word2VecHyperparameters, Word2VecModel
from witchcraft.ml.models.glove import GloVeModel, GloVeHyperparameters
from witchcraft.ml.optimizers import WitchcraftAdagradOptimizer
from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from projects.naughty.definition import UrbanDictionaryDefinition
from witchcraft.nlp.datatypes import Document, Corpus


def corpus_from_files(files: List[str]):
    docs: List[Document] = []
    for filename in files:
        print("Loading file: " + filename)
        with open(filename, 'rb') as fin:
            for pb_str in protobufs_from_filestream(fin):
                pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
                definition = UrbanDictionaryDefinition.from_protobuf(pb)
                docs += [definition.get_word(), definition.get_definition()]

    return Corpus(docs)

# hyperparameters = Word2VecHyperparameters()\
#     .set_max_vocab_size(45000)\
#     .set_min_word_count(10)\
#     .set_optimizer(WitchcraftAdagradOptimizer(learning_rate=0.75))\
#     .set_batch_size(64)\
#     .set_negative_sample_count(64)\
#     .enable_ngram(min_count=30, max_ngram_size=4)\
#     .set_name("win")
#
# vocab: Optional[Word2VecVocab] = Word2VecVocab.load_metadata_from_disk(hyperparameters=hyperparameters)

# if vocab is None:
#     print ("Vocab hasn't been built yet. Building.")
#     p = Pool(20)
#     vocab_builder: Word2VecVocabBuilder = Word2VecVocabBuilder()
#
#     vocab = vocab_builder.build_and_save(
#         corpus=corpus_from_files(argv[1:]),
#         hyperparameters=hyperparameters
#     )
#
#     i = 0
#
#
# print("Done... loading model.")
# model: Word2VecModel = Word2VecModel(
#     vocab=vocab,
#     hyperparameters=hyperparameters
# )

# for filename in argv[1:]:
#     with open(filename, 'rb') as fin:
#         for pb_str in protobufs_from_filestream(fin):
#             pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
#             definition = UrbanDictionaryDefinition.from_protobuf(pb)
#             vocab.get_skipgrams_for_sequence(definition.get_definition_sequence())


print("Building")

hyperparameters: GloVeHyperparameters = GloVeHyperparameters()\
    .set_name("glove_test_const")\
    .set_embedding_size(300)\
    .set_batch_size(1000)\
    .set_loss_weight_alpha(0.75)\
    .set_loss_weight_xmax(100)\
    .set_min_word_count(20)\
    .set_max_vocab_size(30000)\
    .set_optimizer(WitchcraftAdagradOptimizer(1.0))\
    .set_distance_weight_function(lambda d: 1.0 / (1+abs(d)))

model: GloVeModel = GloVeModel(corpus=corpus_from_files(argv[1:]), hyperparameters=hyperparameters)

print("Training")
i = 0
while True:
    model.train(i)
    i += 1

    if i % 500 == 0:
        print (str(i))
        model.save_embeddings("glove_test_const_" + str(i) + ".embeddings")
