from typing import Optional

import tensorflow as tf

from witchcraft.ml.optimizers import WitchcraftAdagradOptimizer, Optimizer
from witchcraft.ml.datasets import WitchcraftDataset
from witchcraft.nlp.datatypes import Corpus
from witchcraft.ml.datatypes import PhraseEmbedding
from witchcraft.util.protobuf import protobuf_to_filestream

class FastTextHyperparameters:
    def __init__(self) -> None:
        self._embedding_size: int = 300
        self._batch_size: int = 64
        self._window_size: int = 4
        self._subsample_threshold: Optional[float] = None
        self._optimizer: Optimizer = WitchcraftAdagradOptimizer(learning_rate=1.5)
        self._name: Optional[str] = None

    def set_embedding_size(self, embedding_size: int) -> 'FastTextHyperparameters':
        self._embedding_size = embedding_size
        return self

    def get_embedding_size(self) -> int:
        return self._embedding_size

    def set_batch_size(self, batch_size: int) -> 'FastTextHyperparameters':
        self._batch_size = batch_size
        return self

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_name(self) -> str:
        return self._name if self._name is not None else "<default>"

    def set_name(self, name) -> 'FastTextHyperparameters':
        self._name = name
        return self

    def set_skipgram_window_size(self, window_size: int) -> 'FastTextHyperparameters':
        self._window_size = window_size
        return self

    def get_skipgram_window_size(self) -> int:
        return self._window_size

    def set_optimizer(self, optimizer: Optimizer) -> 'FastTextHyperparameters':
        self._optimizer = optimizer
        return self

    def get_optimizer(self) -> Optimizer:
        return self._optimizer


allowed_chars = 'abcdefghijklmnopqrstuvwxyz-\''
ngram_id_lookup = {}

i = 1 # i = 0 is a nil padding character
for x in allowed_chars:
    for y in allowed_chars:
        ngram_id_lookup["<" + x + y] = i
        i += 1

        ngram_id_lookup[x + y + ">"] = i
        i += 1

        for z in allowed_chars:
            ngram_id_lookup[x+y+z] = i
            i += 1

print(ngram_id_lookup)

TOTAL_NGRAM_EMBEDDINGS = len(ngram_id_lookup) + 1 # +1 for nil (0)
MAX_VOCAB_SIZE = 40000
MIN_WORD_APPEARANCE = 10
MAX_PHRASE_LEN = 32
NGRAM_LENGTH = 3


class FastTextVocab:
    def __init__(self, corpus: Corpus) -> None:
        i = 0
        self._vocab_counts = {}
        for document in corpus:
            i += 1
            if i % 1000 == 0:
                print("DOC " + str(i))

            for sentence in document:
                for phrase in sentence:
                    if not phrase.provides_contextual_value():
                        continue

                    phrase_norm = phrase.to_phrase_normalized()
                    if phrase_norm not in self._vocab_counts:
                        self._vocab_counts[phrase_norm] = 0

                    self._vocab_counts[phrase_norm] += 1

        self._vocab_counts = [(w, c) for w, c in self._vocab_counts.items() if c >= MIN_WORD_APPEARANCE]
        self._vocab_counts = sorted(self._vocab_counts, key=lambda a: -a[1])
        self._vocab_counts = self._vocab_counts[:MAX_VOCAB_SIZE]
        self._vocab_size = len(self._vocab_counts)
        self._vocab_to_id = {}
        print(self._vocab_counts[:20])

        i = 0
        for (w, c) in self._vocab_counts:
            self._vocab_to_id[w] = i
            i += 1

    def __iter__(self):
        return iter(self._vocab_counts)

    def phrase_to_id(self, phrase):
        return self._vocab_to_id[phrase] if phrase in self._vocab_to_id else None

    def get_vocab_size(self):
        return self._vocab_size

    def create_phrase_ngram_ids(self, phrase):
        phrase = "<" + phrase + ">"
        ngrams = []
        #chunk into ngrams.
        for i in range(len(phrase) - NGRAM_LENGTH + 1):
            ngrams += [phrase[i:i+NGRAM_LENGTH]]

        ngram_ids = [ngram_id_lookup[n] for n in ngrams if n in ngram_id_lookup]
        if len(ngrams) != len(ngram_ids):
            return None

        #Concat/Expand to match correct shape
        ngram_ids = ngram_ids[:MAX_PHRASE_LEN]
        ngram_ids += [0] * (MAX_PHRASE_LEN - len(ngrams))

        return ngram_ids


class FastTextCorpusDataset(WitchcraftDataset):
    def __init__(self, corpus: Corpus, vocab: FastTextVocab) -> None:
        def gen():
            for document in corpus:
                current_doc_ngramid_set = {}
                for sentence in document:
                    for phrase in sentence:
                        phrase_norm = phrase.to_phrase_normalized()
                        if phrase_norm not in current_doc_ngramid_set:
                            ngram_ids = vocab.create_phrase_ngram_ids(phrase_norm)
                            if ngram_ids is None:
                                continue

                            current_doc_ngramid_set[phrase_norm] = ngram_ids

                for sentence in document:
                    target_map = {}
                    for (target, context, distance) in sentence.skipgrams(5):
                        if distance == 0:
                            continue

                        context_norm = context.to_phrase_normalized()
                        target_norm = target.to_phrase_normalized()

                        if context_norm == target_norm:
                            continue

                        context_id = vocab.phrase_to_id(context_norm)
                        target_id = vocab.phrase_to_id(target_norm)

                        if context_id is None or target_id is None or target_norm not in current_doc_ngramid_set:
                            continue

                        if target_id not in target_map:
                            target_map[target_id] = {
                                'context_ids': [],
                                'norm': target_norm
                            }

                        target_map[target_id]['context_ids'] += [context_id]

                    for target_id, metadata in target_map.items():
                        target = current_doc_ngramid_set[metadata['norm']]
                        yield (target, target_id, (metadata['context_ids'] * 5)[:5])

        super(FastTextCorpusDataset, self).__init__(tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32, tf.int32), (tf.TensorShape([MAX_PHRASE_LEN]), tf.TensorShape([]), tf.TensorShape([5]))))


class FastTextModel:
    def __init__(self,
                 corpus: Corpus,
                 hyperparameters: Optional[FastTextHyperparameters] = None) -> None:

        if hyperparameters is None:
            hyperparameters = FastTextHyperparameters()

        self._hyperparameters: FastTextHyperparameters = hyperparameters
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        self._corpus = corpus
        self._vocab = FastTextVocab(corpus=corpus)

        with self._graph.as_default():
            self._dataset = FastTextCorpusDataset(corpus, self._vocab)
            self._dataset = self._dataset.repeat()
            self._dataset = self._dataset.shuffle(shuffle_buffer=100000)
            self._dataset = self._dataset.batch(batch_size=self._hyperparameters.get_batch_size())
            self._dataset = self._dataset.prefetch(buffer_size=100000)

            (target_phrase_ngram_ids, target_phrase_label, context_phrase_labels) = self._dataset.to_tf_iterator().get_next()

            target_phrase_ngram_ids = tf.reshape(target_phrase_ngram_ids, [self._hyperparameters.get_batch_size(), MAX_PHRASE_LEN])
            # target_phrase_label = tf.reshape(target_phrase_label, [self._hyperparameters.get_batch_size()])
            # context_phrase_label = tf.reshape(context_phrase_label, [self._hyperparameters.get_batch_size()])

            print(target_phrase_ngram_ids.shape)
            print(target_phrase_label.shape)
            print(context_phrase_labels.shape)

            # context_phrase_label = tf.expand_dims(context_phrase_label, axis=-1)
            embedding_size = self._hyperparameters.get_embedding_size()

            self._ngram_embeddings = tf.Variable(
                tf.random_uniform([TOTAL_NGRAM_EMBEDDINGS, self._hyperparameters.get_embedding_size()], -1.0, 1.0),
            )

            self._phrase_embeddings = tf.Variable(
                tf.random_uniform([self._vocab.get_vocab_size(), self._hyperparameters.get_embedding_size()], -1.0, 1.0),
            )

            def build_phrase_vector(ngram_batch, phrase_label_batch):
                with tf.variable_scope("phrase_vector", reuse=tf.AUTO_REUSE):
                    #Right now, this just a bag-of-means on the ngrams... cool stuff next if this works.
                    current_phrase_ngram_embeddings = tf.gather(self._ngram_embeddings, ngram_batch)
                    embedding_mask = tf.maximum(tf.sign(ngram_batch), 0) #Zero ids = 0, All non zero ids = 1. To mask out nils.
                    phrase_lens = tf.reduce_sum(embedding_mask, axis=-1)
                    gru_cell = tf.nn.rnn_cell.GRUCell(embedding_size)
                    gru_outputs, gru_state = tf.nn.dynamic_rnn(gru_cell, current_phrase_ngram_embeddings, sequence_length=phrase_lens, dtype=tf.float32)

                    #3 layer NN on the RNN outputs
                    attention = tf.layers.dense(gru_outputs, embedding_size)
                    attention = tf.layers.dense(attention, embedding_size)
                    attention = tf.layers.dense(attention, 1)

                    #Masked softmax
                    attention = tf.exp(attention)
                    attention = attention * tf.cast(tf.expand_dims(embedding_mask, axis=-1), tf.float32)
                    attention_sum = tf.reduce_sum(attention, axis=-1, keepdims=True)
                    attention_sum = tf.maximum(attention_sum, 1e-9)
                    attention = attention / attention_sum

                    return tf.reduce_sum(current_phrase_ngram_embeddings * attention, axis=-2)


                    # phrase_embedding = tf.gather(self._phrase_embeddings, phrase_label_batch)
                    # return phrase_embedding #tf.concat(phrase_by_ngram_embedding, phrase_embedding)


            target_phrase_embeddings = build_phrase_vector(target_phrase_ngram_ids, target_phrase_label)

            #Used for evaluating
            # self._placeholder_phrase_ngram_ids = tf.placeholder(shape=[MAX_PHRASE_LEN], dtype=tf.int32)
            # self._gen_embedding = build_phrase_vector(tf.expand_dims(self._placeholder_phrase_ngram_ids, axis=0))
            # self._gen_embedding = tf.squeeze(self._gen_embedding)

            # context_phrase_embeddings += tf.gather(self._vocab_embedding_matrix, target_phrase_label)

            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [self._vocab.get_vocab_size(), embedding_size],
                    stddev=1.0 / embedding_size ** 0.5
                ),
                name="NoiseConstrastiveWeights"
            )

            nce_bias = tf.Variable(tf.zeros([self._vocab.get_vocab_size()]), name="NoiseConstrastiveBiases")


            self._nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_bias,
                    labels=context_phrase_labels,
                    inputs=target_phrase_embeddings,
                    num_sampled=64,
                    num_true=5,
                    num_classes=self._vocab.get_vocab_size()
                ),
                name="MeanNCELoss"
            )

            self._loss = self._nce_loss


            # print ("LOSS SHAPE: " + str(self._loss.shape))

            self._optimizer = hyperparameters.get_optimizer().to_tf_optimizer()
            self._minimize = self._optimizer.minimize(self._loss)

            # grads = tf.gradients(self._loss, tf.trainable_variables())
            # grads, _ = tf.clip_by_global_norm(grads, 0.5)  # gradient clipping
            # grads_and_vars = list(zip(grads, tf.trainable_variables()))
            # self._minimize = self._optimizer.apply_gradients(grads_and_vars)

            self._summary_loss = tf.summary.scalar("loss", self._loss)
            self._summary = tf.summary.merge([ self._summary_loss])
            self._writer = tf.summary.FileWriter('./logs/' + self._hyperparameters.get_name(), self._session.graph)
            self._session.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

    def train(self, global_step: int) -> None:
        with self._graph.as_default():
            _, calc_summary = self._session.run([self._minimize, self._summary])
            self._writer.add_summary(calc_summary, global_step=global_step)

            if global_step % 1000 == 0:
                self._saver.save(self._session, './logs/' + self._hyperparameters.get_name() + '.ckpt', global_step)

    def save_embeddings(self, filename: str) -> None:
        print("Starting to save...")

        # phrase_embeddings = list(self._session.run(self._phrase_embeddings))
        # with open(filename, 'wb') as fout:
        #     i = 0
        #     for (phrase, count) in self._vocab:
        #         embedding_vector = phrase_embeddings[self._vocab.phrase_to_id(phrase)]
        #         embedding_proto = PhraseEmbedding(phrase, count, embedding_vector).to_protobuf()
        #         protobuf_to_filestream(fout, embedding_proto.SerializeToString())
        #         i += 1

        print("Done saving")

