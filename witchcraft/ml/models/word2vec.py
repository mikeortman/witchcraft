from typing import Optional, Dict, List, Tuple, Generator
import tensorflow as tf
import os
from pathlib import Path

from witchcraft.ml.optimizers import Optimizer
from witchcraft.ml.datasets import WitchcraftDataset
from witchcraft.nlp.protos.nlpdatatypes_pb2 import WordEmbedding as WordEmbeddingProto
from witchcraft.util.protobuf import protobuf_to_filestream

class Word2VecHyperparameters:
    def __init__(self) -> None:
        self._embedding_size: int = 300
        self._batch_size: int = 64
        self._negative_sample_count: int = 64
        self._min_word_count: int = 20
        self._max_vocab_size: Optional[int] = None
        self._window_size: int = 3
        self._subsample_threshold: Optional[float] = None
        self._optimizer: Optimizer = Optimizer()
        self._name: Optional[str] = None
        self._record_file_size: int = 100000

    def set_embedding_size(self, embedding_size: int) -> 'Word2VecHyperparameters':
        self._embedding_size = embedding_size
        return self

    def get_embedding_size(self) -> int:
        return self._embedding_size

    def set_batch_size(self, batch_size: int) -> 'Word2VecHyperparameters':
        self._batch_size = batch_size
        return self

    def get_batch_size(self) -> int:
        return self._batch_size

    def set_negative_sample_count(self, negative_sample_count: int) -> 'Word2VecHyperparameters':
        self._negative_sample_count = negative_sample_count
        return self

    def get_negative_sample_count(self) -> int:
        return self._negative_sample_count

    def get_name(self) -> str:
        return self._name if self._name is not None else "<default>"

    def set_name(self, name) -> 'Word2VecHyperparameters':
        self._name = name
        return self

    def set_min_word_count(self, min_word_count: int) -> 'Word2VecHyperparameters':
        self._min_word_count = min_word_count
        return self

    def get_min_word_count(self) -> Optional[int]:
        return self._min_word_count

    def set_max_vocab_size(self, max_vocab_size: Optional[int]) -> 'Word2VecHyperparameters':
        self._max_vocab_size = max_vocab_size
        return self

    def get_max_vocab_size(self) -> Optional[int]:
        return self._max_vocab_size

    def set_skipgram_window_size(self, window_size: int) -> 'Word2VecHyperparameters':
        self._window_size = window_size
        return self

    def get_skipgram_window_size(self) -> int:
        return self._window_size

    def set_subsampling_threshold(self, subsample_threshold: Optional[float]) -> 'Word2VecHyperparameters':
        self._subsample_threshold = subsample_threshold
        return self

    def get_subsampling_threshold(self) -> Optional[float]:
        return self._subsample_threshold

    def set_optimizer(self, optimizer: Optimizer) -> 'Word2VecHyperparameters':
        self._optimizer = optimizer
        return self

    def get_optimizer(self) -> Optimizer:
        return self._optimizer

    def set_record_file_size(self, record_file_size: int) -> 'Word2VecHyperparameters':
        self._record_file_size = record_file_size
        return self

    def get_record_file_size(self) -> int:
        return self._record_file_size


class Word2VecVocab:
    def __init__(self, hyperparameters: Optional[Word2VecHyperparameters],
                 pruned: Dict[str, int],
                 pruned_word_to_id: Dict[str, int],
                 pruned_id_to_word: List[str]) -> None:
        if hyperparameters is None:
            hyperparameters = Word2VecHyperparameters()

        self._hyperparameters: Word2VecHyperparameters = hyperparameters
        self._pruned: Dict[str, int] = pruned
        self._pruned_word_to_id: Dict[str, int] = pruned_word_to_id
        self._pruned_id_to_word: List[str] = pruned_id_to_word
        self._vocab_size: int = len(self._pruned_id_to_word)

    def word_to_id(self, word: str) -> int:
        if word not in self._pruned_word_to_id:
            return None

        return self._pruned_word_to_id[word]

    def id_to_word(self, id: int) -> Optional[str]:
        if id not in self._pruned_id_to_word:
            return None

        return self._pruned_id_to_word[id]

    def get_vocab_size(self):
        return self._vocab_size

    def get_vocab(self):
        return self._pruned

    def get_word_list(self):
        return self._pruned_id_to_word

    def store_skipgrams_to_disk(self, seq_gen) -> None:
        current_writer_index = 0
        i = 0
        current_writer = tf.python_io.TFRecordWriter(self.get_skipgram_record_filename(current_writer_index))

        def generate_skipgrams(sentence: List[Optional[int]]) -> Generator[Tuple[int, int], None, None]:
            sentence_size: int = len(sentence)
            window_size: int = self._hyperparameters.get_skipgram_window_size()

            for i in range(sentence_size):
                current_word = sentence[i]
                if current_word is None:
                    continue

                for y in range(1, window_size + 1):
                    if i - y >= 0 and sentence[i - y] is not None:
                        yield (current_word, sentence[i - y])

                for y in range(1, window_size + 1):
                    if i + y < sentence_size and sentence[i + y] is not None:
                            yield (current_word, sentence[i + y])

        for seq in seq_gen():
            for sentence in seq.get_sentence_generator():
                words = [w.get_word_string_normalized() for w in sentence.get_word_generator() if w.provides_contextual_value()]
                words = [self.word_to_id(w) for w in words]
                for (skipgram_source, skipgram_target) in generate_skipgrams(words):
                    skipgrams_source_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[skipgram_source]))
                    skipgrams_target_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[skipgram_target]))

                    features = {
                        's': skipgrams_source_feature,
                        't': skipgrams_target_feature
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    current_writer.write(example.SerializeToString())

                    i += 1
                    if i % self._hyperparameters.get_record_file_size() == 0:
                        print ("New file: " + str(i))
                        current_writer.close()
                        current_writer_index += 1
                        current_writer = tf.python_io.TFRecordWriter(self.get_skipgram_record_filename(current_writer_index))


    def load_skipgrams_to_dataset(self) -> 'Word2VecSkipgramDataset':
        return Word2VecSkipgramDataset(self)

    def save_metadata_to_disk(self) -> None:
        with open(Word2VecVocab.get_metadata_filename(self._hyperparameters), "w") as fout:
            fout.write("word\tword_count\n")
            for word in self._pruned_id_to_word:
                fout.write(word + "\t" + str(self._pruned[word]) + "\n")

    @classmethod
    def get_metadata_filename(cls, hyperparameters: Optional[Word2VecHyperparameters]) -> str:
        if hyperparameters is None:
            hyperparameters = Word2VecHyperparameters()

        return "words_" + hyperparameters.get_name() + ".tsv"

    def get_skipgram_record_filename(self, idx: int = 0) -> str:
        return "skipgrams_" + self._hyperparameters.get_name() + "_" + str(idx) + ".tfrecord"

    def get_skipgram_record_filenames(self):
        # Build a list of strings containing all valid filenames matching the record filename pattern
        known_filenames = []
        i = 0
        while True:
            current_test_filename = self.get_skipgram_record_filename(i)
            if not os.path.isfile(current_test_filename):
                return known_filenames

            known_filenames += [current_test_filename]
            i += 1



    def to_dataset(self) -> 'Word2VecSkipgramDataset':
        return Word2VecSkipgramDataset(self)

    @classmethod
    def load_metadata_from_disk(cls, hyperparameters: Optional[Word2VecHyperparameters]) -> Optional['Word2VecVocab']:
        filename = Word2VecVocab.get_metadata_filename(hyperparameters)
        if not Path(filename).is_file():
            return None

        pruned = {}
        word_to_id = {}
        id_to_word = []

        with open(Word2VecVocab.get_metadata_filename(hyperparameters), "r") as fin:
            i = 0
            on_column_row = True
            for line in fin:
                if on_column_row == True:
                    on_column_row = False
                    continue # Skip the header

                row = line.split('\t')
                word = row[0]
                word_count = int(row[1])
                pruned[word] = word_count

                word_to_id[word] = i
                id_to_word += [word]
                i += 1

        return Word2VecVocab(
            hyperparameters=hyperparameters,
            pruned=pruned,
            pruned_word_to_id=word_to_id,
            pruned_id_to_word=id_to_word
        )


class Word2VecSkipgramDataset(WitchcraftDataset):
    def __init__(self, vocab: Word2VecVocab) -> None:
        super(Word2VecSkipgramDataset, self).__init__(tf.data.TFRecordDataset(vocab.get_skipgram_record_filenames()))
        self._vocab = vocab

        def map_entry(entry_proto):
            features = {
                's': tf.FixedLenFeature([], tf.int64),
                't': tf.FixedLenFeature([], tf.int64)
            }
            parsed_features = tf.parse_single_example(entry_proto, features)
            return parsed_features["s"], parsed_features["t"]

        self.map(map_entry, num_parallel_calls=50)

    def get_vocab(self):
        return self._vocab


class Word2VecVocabBuilder:
    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._last_built_vocab: Optional[Word2VecVocab] = None

    def add_word_count_to_vocab(self, word: str, count: int) -> None:
        if word not in self._vocab:
            self._vocab[word] = 0

        self._vocab[word] += count
        self._last_built_vocab = None #Invalidate the build vocab

    def build(self, hyperparameters: Optional[Word2VecHyperparameters] = None) -> Word2VecVocab:
        if hyperparameters is None:
            hyperparameters = Word2VecHyperparameters()

        if self._last_built_vocab is not None:
            return self._last_built_vocab

        pruned = [(k,v) for k,v in self._vocab.items() if v >= hyperparameters.get_min_word_count()]
        pruned.sort(key=lambda x: -x[1])



        # Subsampling
        # if hyperparameters.get_subsampling_threshold() is not None:
        #     log.info("Subsampling enabled. Calculating corpus size...")
        #     total_tokens = 0
        #     for token, count in tokenCounts:
        #         total_tokens += count
        #
        #     log.info("Total tokens: " + str(totalTokensInCorpus) + ". Performing subsampling...")
        #     tokensToKeep = []
        #     for token, count in tokenCounts:
        #         token_freq = count / totalTokensInCorpus
        #         if token_freq >= hyperparameters.get_subsampling_threshold():
        #             log.info("Removing '" + token + "' during subsampling; frequency: " + str(token_freq * 100) + "%")
        #             continue
        #
        #         tokensToKeep += [(token, count)]
        #
        #     tokenCounts = tokensToKeep
        # else:
        #     log.info("Subsampling disabled.")


        max_vocab_size = hyperparameters.get_max_vocab_size()
        if max_vocab_size is not None:
            pruned = pruned[:max_vocab_size]

        pruned_word_to_id = {}
        pruned_id_to_word = []

        i = 0
        for (word, word_count) in pruned:
            pruned_word_to_id[word] = i
            pruned_id_to_word += [word]
            i += 1

        pruned = {k: v for (k, v) in pruned}

        self._last_built_vocab = Word2VecVocab(
            hyperparameters=hyperparameters,
            pruned=pruned,
            pruned_word_to_id=pruned_word_to_id,
            pruned_id_to_word=pruned_id_to_word
        )

        return self._last_built_vocab


class Word2VecModel:
    # Classic word2vec model for use with witchcraft
    def __init__(self,
                 vocab: Word2VecVocab,
                 hyperparameters: Optional[Word2VecHyperparameters] = None) -> None:

        if hyperparameters is None:
            hyperparameters = Word2VecHyperparameters()

        self._hyperparameters: Word2VecHyperparameters = hyperparameters
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        self._vocab = vocab

        with self._graph.as_default():
            # self._dataset = tf.data.Dataset.from_generator(
            #     training_pair_generator,
            #     (tf.int32, tf.int32),
            #     output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
            # )

            self._dataset = vocab.to_dataset()
            # self._dataset = self._dataset.cache()
            self._dataset = self._dataset.repeat()
            self._dataset = self._dataset.shuffle(shuffle_buffer=1000000)
            self._dataset = self._dataset.batch(batch_size=self._hyperparameters.get_batch_size())
            self._dataset = self._dataset.prefetch(buffer_size=1000000)

            (center_words, target_words) = self._dataset.to_tf_iterator().get_next()
            print(center_words.shape)
            print(target_words.shape)
            target_words = tf.expand_dims(target_words, axis=-1)

            embedding_size = self._hyperparameters.get_embedding_size()
            self._word_embed_matrix = tf.Variable(
                tf.random_uniform([self._vocab.get_vocab_size(), embedding_size], -1.0, 1.0),
                name="WordEmbeddingsMatrix"
            )

            center_embeddings = tf.nn.embedding_lookup(self._word_embed_matrix, center_words, name="WordEmbeddings")

            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [self._vocab.get_vocab_size(), embedding_size],
                    stddev=1.0 / embedding_size ** 0.5
                ),
                name="NoiseConstrastiveWeights"
            )

            nce_bias = tf.Variable(tf.zeros([self._vocab.get_vocab_size()]), name="NoiseConstrastiveBiases")

            self._loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_bias,
                    labels=target_words,
                    inputs=center_embeddings,
                    num_sampled=self._hyperparameters.get_negative_sample_count(),
                    num_classes=self._vocab.get_vocab_size()
                ),
                name="MeanNCELoss"
            )

            self._optimizer = hyperparameters.get_optimizer().to_tf_optimizer().minimize(self._loss)
            self._summary = tf.summary.scalar("loss", self._loss)
            self._writer = tf.summary.FileWriter('./logs/' + self._hyperparameters.get_name(), self._session.graph)
            self._session.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

    def train(self, global_step):
        with self._graph.as_default():
            _, calc_summary = self._session.run([self._optimizer, self._summary])
            self._writer.add_summary(calc_summary, global_step=global_step)

            if global_step % 1000 == 0:
                self._saver.save(self._session, './logs/' + self._hyperparameters.get_name() + '.ckpt', global_step)

    def save_embeddings(self, filename):
        print("Starting to save...")
        with open(filename, 'wb') as fout:
            vocab_list = self._vocab.get_word_list()
            embeddings = self._session.run(self._word_embed_matrix)
            for i in range(len(vocab_list)):
                word = vocab_list[i]
                embedding = embeddings[i]
                # print(word + ": " + str(embedding))
                embeddingProto = WordEmbeddingProto(
                    word=word,
                    embeddingVector=embedding
                )
                protobuf_to_filestream(fout, embeddingProto.SerializeToString())

        print("Done saving")

