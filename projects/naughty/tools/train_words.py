from sys import argv
from multiprocessing import Pool
from typing import List, Dict, Optional
import itertools

from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.ml.models.word2vec import Word2VecVocabBuilder, Word2VecVocab, Word2VecHyperparameters, Word2VecModel
from witchcraft.ml.optimizers import WitchcraftAdagradOptimizer
from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from projects.naughty.definition import UrbanDictionaryDefinition
from witchcraft.nlp.parse import Parser


hyperparameters = Word2VecHyperparameters()\
    .set_max_vocab_size(35000)\
    .set_min_word_count(10)\
    .set_optimizer(WitchcraftAdagradOptimizer(learning_rate=0.75))\
    .set_batch_size(64)\
    .set_negative_sample_count(64)\
    .enable_ngram(min_count=30, max_ngram_size=4)\
    .set_name("win")

vocab: Optional[Word2VecVocab] = Word2VecVocab.load_metadata_from_disk(hyperparameters=hyperparameters)



if vocab is None:
    print ("Vocab hasn't been built yet. Building.")
    p = Pool(20)
    vocab_builder: Word2VecVocabBuilder = Word2VecVocabBuilder()


    def sentence_sequence_generator_from_files():
        for filename in argv[1:]:
            print("Loading file: " + filename)
            with open(filename, 'rb') as fin:
                for pb_str in protobufs_from_filestream(fin):
                    pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
                    definition = UrbanDictionaryDefinition.from_protobuf(pb)
                    yield definition.get_word_sequence()
                    yield definition.get_definition_sequence()

    vocab = vocab_builder.build_and_save(
        sentence_sequence_generator=sentence_sequence_generator_from_files,
        hyperparameters=hyperparameters
    )

    i = 0


print("Done... loading model.")
model: Word2VecModel = Word2VecModel(
    vocab=vocab,
    hyperparameters=hyperparameters
)

print("Training")
i = 0
while True:
    model.train(i)
    i += 1

    if i % 1000 == 0:
        print (str(i * 128))

    if i % 100000 == 0:
        model.save_embeddings("win_" + str(i) + ".embeddings")

# for filename in argv[1:]:
#     with open(filename, 'rb') as fin:
#         for pb_str in protobufs_from_filestream(fin):
#             pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
#             definition = UrbanDictionaryDefinition.from_protobuf(pb)
#             vocab.get_skipgrams_for_sequence(definition.get_definition_sequence())