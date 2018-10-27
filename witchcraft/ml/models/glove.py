import numpy as np
import tensorflow as tf

from typing import Optional, Callable
from witchcraft.nlp.datatypes import Corpus
from witchcraft.ml.datasets import WitchcraftDatasetIntegerRange
from witchcraft.ml.optimizers import Optimizer
from witchcraft.ml.datatypes import PhraseEmbedding
from witchcraft.util.protobuf import protobuf_to_filestream

class GloVeHyperparameters:
    def __init__(self) -> None:
        self._embedding_size: int = 300
        self._batch_size: int = 1000
        self._min_word_count: int = 20
        self._max_vocab_size: Optional[int] = None
        self._window_size: int = 3
        self._optimizer: Optimizer = Optimizer()
        self._name: Optional[str] = None
        self._distance_func: Callable[[], float] = lambda d: 1.0 / abs(d)
        self._loss_weight_alpha: float = 0.75
        self._loss_weight_xmax: float = 100


    def set_embedding_size(self, embedding_size: int) -> 'GloVeHyperparameters':
        self._embedding_size = embedding_size
        return self

    def get_embedding_size(self) -> int:
        return self._embedding_size

    def set_batch_size(self, batch_size: int) -> 'GloVeHyperparameters':
        self._batch_size = batch_size
        return self

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_name(self) -> str:
        return self._name if self._name is not None else "<default>"

    def set_name(self, name) -> 'GloVeHyperparameters':
        self._name = name
        return self

    def set_min_word_count(self, min_word_count: int) -> 'GloVeHyperparameters':
        self._min_word_count = min_word_count
        return self

    def get_min_word_count(self) -> Optional[int]:
        return self._min_word_count

    def set_max_vocab_size(self, max_vocab_size: Optional[int]) -> 'GloVeHyperparameters':
        self._max_vocab_size = max_vocab_size
        return self

    def get_max_vocab_size(self) -> Optional[int]:
        return self._max_vocab_size

    def set_window_size(self, window_size: int) -> 'GloVeHyperparameters':
        self._window_size = window_size
        return self

    def get_window_size(self) -> int:
        return self._window_size

    def set_distance_weight_function(self, distance_func: Callable[[float], float]) -> 'GloVeHyperparameters':
        self._distance_func = distance_func
        return self

    def distance_weight(self, weight: float) -> float:
        return self._distance_func(weight)

    def set_optimizer(self, optimizer: Optimizer) -> 'GloVeHyperparameters':
        self._optimizer = optimizer
        return self

    def get_optimizer(self) -> Optimizer:
        return self._optimizer

    def set_loss_weight_alpha(self, loss_weight_alpha: float) -> 'GloVeHyperparameters':
        self._loss_weight_alpha = loss_weight_alpha
        return self

    def get_loss_weight_alpha(self) -> float:
        return self._loss_weight_alpha

    def set_loss_weight_xmax(self, loss_weight_xmax: float) -> 'GloVeHyperparameters':
        self._loss_weight_xmax = loss_weight_xmax
        return self

    def get_loss_weight_xmax(self) -> float:
        return self._loss_weight_xmax

class GloVeModel:
    def __init__(self, corpus: Corpus, hyperparameters: Optional[GloVeHyperparameters]) -> None:
        self._train_loss_accum_count:int = 0
        self._train_loss_accum:float = 0.0

        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

        self._hyperparameters = hyperparameters
        if self._hyperparameters is None:
            self._hyperparameters = GloVeHyperparameters()

        print ("Calculating phrase counts...")
        phrase_counts = {}
        for document in corpus:
            for sentence in document:
                for phrase in sentence:
                    if not phrase.provides_contextual_value():
                        continue

                    phrase_norm = phrase.to_phrase_normalized()
                    if phrase_norm not in phrase_counts:
                        phrase_counts[phrase_norm] = 0

                    phrase_counts[phrase_norm] += 1

        print ("Sorting by size...")
        self._phrase_counts = [(p, c) for p, c in phrase_counts.items() if c >= self._hyperparameters.get_min_word_count()]
        sorted(self._phrase_counts, key=lambda a: -a[1])

        print ("Truncating...")
        self._phrase_counts = self._phrase_counts[:self._hyperparameters.get_max_vocab_size()]

        total_phrases = len(self._phrase_counts)
        print ("Total phrases: " + str(total_phrases))

        i = 0
        self._phrase_id_map = {}
        for (phrase, count) in self._phrase_counts:
            self._phrase_id_map[phrase] = i
            i += 1

        print ("Initializing coocurrance matrix")
        self._cooccurance_matrix_arr = np.empty([total_phrases, total_phrases], dtype=np.float32)

        print ("Building coocurrance matrix")
        i = 0
        for document in corpus:
            for sentence in document:
                for (source, target, distance) in sentence.skipgrams(self._hyperparameters.get_window_size()):

                    source_norm = source.to_phrase_normalized()
                    target_norm = target.to_phrase_normalized()

                    if source_norm not in self._phrase_id_map or target_norm not in self._phrase_id_map:
                        continue

                    source_idx = self._phrase_id_map[source.to_phrase_normalized()]
                    target_idx = self._phrase_id_map[target.to_phrase_normalized()]
                    self._cooccurance_matrix_arr[source_idx, target_idx] += self._hyperparameters.distance_weight(distance)

            i += 1
            if i % 1000 == 0:
                print ("Done with document " + str(i))



        print("Building TF graph")
        with self._graph.as_default():

            self._cooccurance_matrix_placeholder = tf.placeholder(dtype=tf.float32, shape=[total_phrases, total_phrases])
            self._cooccurance_matrix = tf.Variable(self._cooccurance_matrix_placeholder, trainable=False)

            batch_size = self._hyperparameters.get_batch_size()
            target_word_id_dataset = WitchcraftDatasetIntegerRange(total_phrases)\
                .repeat()\
                .shuffle(total_phrases*2)\
                .batch(batch_size)\
                .prefetch(total_phrases*2)

            context_word_id_dataset = WitchcraftDatasetIntegerRange(total_phrases)\
                .repeat()\
                .shuffle(total_phrases*2)\
                .batch(batch_size)\
                .prefetch(total_phrases*2)



            #Build the model
            self._word_embeddings_target = tf.Variable(
                tf.random_uniform([total_phrases, self._hyperparameters.get_embedding_size()], -1.0, 1.0),
                name="WordEmbeddingsMatrixTarget"
            )

            self._word_embeddings_context = tf.Variable(
                tf.random_uniform([total_phrases, self._hyperparameters.get_embedding_size()], -1.0, 1.0),
                name="WordEmbeddingsMatriContext"
            )


            target_word_id_iter = target_word_id_dataset.to_tf_iterator()
            context_word_id_iter = context_word_id_dataset.to_tf_iterator()
            target_word_id_batch = target_word_id_iter.get_next()
            context_word_id_batch = context_word_id_iter.get_next()

            i = tf.reshape(tf.tile(target_word_id_batch, [batch_size]), shape=[-1, batch_size])
            j = tf.transpose(tf.reshape(tf.tile(context_word_id_batch, [batch_size]), shape=[-1,batch_size]))
            ij = tf.concat([tf.expand_dims(i, axis=-1), tf.expand_dims(j, axis=-1)], axis=-1)
            x_ij = tf.gather_nd(self._cooccurance_matrix, ij)
            log_x_ij = tf.log(tf.maximum(x_ij,1e-9))


            embedding_target = tf.gather_nd(self._word_embeddings_target, tf.expand_dims(i, -1))
            embedding_context = tf.gather_nd(self._word_embeddings_context, tf.expand_dims(j, -1))
            embedding_dot_product = tf.reduce_sum(embedding_target * embedding_context, axis=-1)


            biases_target = tf.Variable(
                tf.random_uniform([total_phrases], -1.0, 1.0),
                name="BiasTarget"
            )

            biases_context = tf.Variable(
                tf.random_uniform([total_phrases], -1.0, 1.0),
                name="BiasContext"
            )

            bias_target = tf.gather_nd(biases_target, tf.expand_dims(i, -1))
            bias_context = tf.gather_nd(biases_context, tf.expand_dims(j, -1))

            xmax = self._hyperparameters.get_loss_weight_xmax()
            alpha = self._hyperparameters.get_loss_weight_alpha()
            loss_weight = tf.minimum((x_ij / xmax) ** alpha, 1.0)

            print(self._word_embeddings_context.shape)
            print(self._word_embeddings_target.shape)

            self._loss = tf.reduce_sum(loss_weight * ((embedding_dot_product + bias_target + bias_context - log_x_ij) ** 2))
            self._word_embeddings = self._word_embeddings_target + self._word_embeddings_context
            print(self._word_embeddings.shape)

            self._optimizer = self._hyperparameters.get_optimizer().to_tf_optimizer().minimize(self._loss)
            self._summary = tf.summary.scalar("loss", self._loss)
            self._writer = tf.summary.FileWriter('./logs/' + self._hyperparameters.get_name(), self._session.graph)

            self._session.run(tf.global_variables_initializer(), feed_dict={self._cooccurance_matrix_placeholder: self._cooccurance_matrix_arr})
            self._saver = tf.train.Saver()

        print ("Done building graph")

    def train(self, global_step: int) -> None:
        with self._graph.as_default():
            _, calc_summary, loss = self._session.run([self._optimizer, self._summary, self._loss])

            self._writer.add_summary(calc_summary, global_step=global_step)
            self._train_loss_accum_count += 1
            self._train_loss_accum += loss

            if global_step % 50 == 0:
                self._train_loss_accum_count = self._train_loss_accum_count if self._train_loss_accum_count > 0 else 1
                print("AVERAGE LOSS: " + str(self._train_loss_accum / self._train_loss_accum_count))
                self._train_loss_accum_count = 0
                self._train_loss_accum = 0

                self._saver.save(self._session, './logs/' + self._hyperparameters.get_name() + '.ckpt', global_step)

    def save_embeddings(self, filename: str) -> None:
        print("Starting to save...")
        with open(filename, 'wb') as fout:
            embedding_vectors = self._session.run(self._word_embeddings)

            for phrase_id in range(len(self._phrase_counts)):
                (phrase, count) = self._phrase_counts[phrase_id]
                embedding_vector = embedding_vectors[phrase_id]
                embedding_proto = PhraseEmbedding(phrase, count, embedding_vector).to_protobuf()
                protobuf_to_filestream(fout, embedding_proto.SerializeToString())

        print("Done saving")
