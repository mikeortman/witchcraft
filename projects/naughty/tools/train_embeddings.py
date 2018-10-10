from sys import argv
from multiprocessing import Pool
from typing import List, Dict, Optional
import itertools

from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.ml.models.word2vec import Word2VecVocabBuilder, Word2VecVocab, Word2VecHyperparameters, Word2VecModel

from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from projects.naughty.definition import UrbanDictionaryDefinition


from witchcraft.ml.datasets import WitchcraftDatasetFiles


# class NaughtyProtobufDataset(WitchcraftDatasetFiles):
#     def __init__(self, file_pattern, vocab: Word2VecVocab) -> None:
#         super(NaughtyProtobufDataset, self).__init__(file_pattern)
#
#         def read_flatmap(file_name) -> List[any]:
#             print("Get: " + str(file_name))
#             skipgrams = []
#
            with open(file_name, 'rb') as fin:
                for pb_str in protobufs_from_filestream(fin):
                    pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
                    definition = UrbanDictionaryDefinition.from_protobuf(pb)
                    skipgrams += vocab.get_skipgrams_for_sequence(definition.get_definition_sequence())
#
#             return skipgrams
#
#         self.flat_map(read_flatmap)




hyperparameters = Word2VecHyperparameters().set_max_vocab_size(10000).set_min_word_count(10)
vocab: Optional[Word2VecVocab] = Word2VecVocab.load_from_disk(hyperparameters=hyperparameters)

def build_wordcount_list_for_file(filename: str) -> List[Dict[str, int]]:
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

def build_dataset_for_file(filename: str) -> None
    with open(filename, 'rb') as fin:
        outfilename = filename + ".records"
        for pb_str in protobufs_from_filestream(fin):
            pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
            definition = UrbanDictionaryDefinition.from_protobuf(pb)
            vocab


if vocab is None:
    print ("Vocab hasn't been built yet. Building.")
    p = Pool(20)
    vocab_builder: Word2VecVocabBuilder = Word2VecVocabBuilder()
    for result in p.map(build_wordcount_list_for_file, argv[1:]):
        # print(word)
        for word, word_count in result.items():
            vocab_builder.add_word_count_to_vocab(word, word_count)

    vocab = vocab_builder.build(
        hyperparameters=hyperparameters
    )

    vocab.save_to_disk()

    ## Build TFRecords.




# dataset: NaughtyProtobufDataset = NaughtyProtobufDataset(file_pattern="*.protobuf", vocab=vocab)

model: Word2VecModel = Word2VecModel(
    dataset=dataset,
    vocab=vocab,
    hyperparameters=hyperparameters
)

print("Training")
for i in range(100000):
    model.train(i)

# for filename in argv[1:]:
#     with open(filename, 'rb') as fin:
#         for pb_str in protobufs_from_filestream(fin):
#             pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
#             definition = UrbanDictionaryDefinition.from_protobuf(pb)
#             vocab.get_skipgrams_for_sequence(definition.get_definition_sequence())