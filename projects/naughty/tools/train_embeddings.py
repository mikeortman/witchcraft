from sys import argv
from multiprocessing import Pool
from typing import List, Dict
import itertools

from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.ml.models.word2vec import Word2VecVocabBuilder, Word2VecVocab, Word2VecHyperparameters

from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from projects.naughty.definition import UrbanDictionaryDefinition


# from witchcraft.ml.datasets import WitchcraftDatasetFiles


# class NaughtyProtobufDataset(WitchcraftDatasetFiles):
#     def __init__(self, file_pattern) -> None:
#         super(NaughtyProtobufDataset, self).__init__(file_pattern)
#
#         def read_flatmap(file_name) -> List[any]:
#             definitions = []
#
#             with open(file_name, 'rb') as fin:
#                 for pb_str in protobufs_from_filestream(fin):
#                     pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
#                     definition = UrbanDictionaryDefinition.from_protobuf(pb)
#                     for sentence definition.get_word_sequence().get_sentence_generator():
#                     definitions += [definition]
#
#             for definition in definitions:
#                 continue
#
#             return definitions
#
#         self.flat_map(read_flatmap)







def read_file(filename: str) -> List[Dict[str, int]]:
    word_counts: Dict[str, int] = {}
    with open(filename, 'rb') as fin:
        for pb_str in protobufs_from_filestream(fin):
            pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
            definition = UrbanDictionaryDefinition.from_protobuf(pb)

            for word in itertools.chain(definition.get_word_sequence().get_word_generator(), definition.get_definition_sequence().get_word_generator()):
                if not word.provides_contextual_value():
                    continue

                word_str = word.get_word_string_normalized()
                if word_str not in word_counts:
                    word_counts[word_str] = 0

                word_counts[word_str] += 1

    return word_counts


p = Pool(20)
vocab_builder: Word2VecVocabBuilder = Word2VecVocabBuilder()
hyperparameters = Word2VecHyperparameters().set_max_vocab_size(10000).set_min_word_count(10)

for result in p.map(read_file, argv[1:]):
    # print(word)
    for word, word_count in result.items():
        vocab_builder.add_word_count_to_vocab(word, word_count)

vocab: Word2VecVocab = vocab_builder.build(
    hyperparameters=hyperparameters
)

print(vocab.get_vocab())

for filename in argv[1:]:
    with open(filename, 'rb') as fin:
        for pb_str in protobufs_from_filestream(fin):
            pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
            definition = UrbanDictionaryDefinition.from_protobuf(pb)
            vocab.get_skipgrams_for_sequence(definition.get_definition_sequence())