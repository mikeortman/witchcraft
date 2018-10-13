from sys import argv
from multiprocessing import Pool
from typing import List, Dict, Optional
import itertools

from witchcraft.util.protobuf import protobufs_from_filestream
from witchcraft.ml.models.word2vec import Word2VecVocabBuilder, Word2VecVocab, Word2VecHyperparameters, Word2VecModel
from witchcraft.ml.optimizers import WitchcraftAdagradOptimizer
from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from projects.naughty.definition import UrbanDictionaryDefinition


hyperparameters = Word2VecHyperparameters()\
    .set_max_vocab_size(25000)\
    .set_min_word_count(15)\
    .set_optimizer(WitchcraftAdagradOptimizer(learning_rate=0.75))\
    .set_batch_size(64)\
    .set_negative_sample_count(64)\
    .set_name("win")

vocab: Optional[Word2VecVocab] = Word2VecVocab.load_metadata_from_disk(hyperparameters=hyperparameters)

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

    i = 0
    vocab.save_metadata_to_disk()


    def gen_seqs():
        i = 0
        for filename in argv[1:]:
            with open(filename, 'rb') as fin:
                print ("Now on: " +str(filename))
                for pb_str in protobufs_from_filestream(fin):
                    pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
                    definition = UrbanDictionaryDefinition.from_protobuf(pb)
                    yield definition.get_definition_sequence()
                    yield definition.get_word_sequence()

                    i += 1
                    if i % 100 == 0:
                        print("Parsed to skipgrams: " + str(i))

    vocab.store_skipgrams_to_disk(gen_seqs)


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