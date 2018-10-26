import numpy as np
import tensorflow as tf

from witchcraft.nlp.datatypes import Corpus
from witchcraft.ml.datasets import WitchcraftDatasetIntegerRange
from witchcraft.ml.optimizers import WitchcraftGradientDescentOptimizer
from witchcraft.ml.datatypes import PhraseEmbedding
from witchcraft.util.protobuf import protobuf_to_filestream


class GloVeModel:
    def __init__(self, corpus: Corpus) -> None:
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

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
        self._phrase_counts = [(phrase, count) for phrase, count in phrase_counts.items()]
        sorted(self._phrase_counts, key=lambda a: -a[1])

        print ("Truncating...")
        self._phrase_counts = self._phrase_counts[:30000]

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
                for (source, target, distance) in sentence.skipgrams(10):
                    source_norm = source.to_phrase_normalized()
                    target_norm = target.to_phrase_normalized()

                    if source_norm not in self._phrase_id_map or target_norm not in self._phrase_id_map:
                        continue

                    source_idx = self._phrase_id_map[source.to_phrase_normalized()]
                    target_idx = self._phrase_id_map[target.to_phrase_normalized()]
                    self._cooccurance_matrix_arr[source_idx, target_idx] += 1.0 / distance

            i += 1
            if i % 1000 == 0:
                print ("Done with document " + str(i))



        print("Building TF graph")
        with self._graph.as_default():
            BATCH_SIZE=500
            self._cooccurance_matrix_placeholder = tf.placeholder(dtype=tf.float32, shape=[total_phrases, total_phrases])
            self._cooccurance_matrix = tf.Variable(self._cooccurance_matrix_placeholder, trainable=False)

            target_word_id_dataset = WitchcraftDatasetIntegerRange(total_phrases).repeat().shuffle(total_phrases*2).batch(BATCH_SIZE).prefetch(total_phrases*2)
            context_word_id_dataset = WitchcraftDatasetIntegerRange(total_phrases).repeat().shuffle(total_phrases*2).batch(BATCH_SIZE).prefetch(total_phrases*2)



            #Build the model
            EMBEDDING_SIZE=300
            self._word_embeddings_target = tf.Variable(
                tf.random_uniform([total_phrases, EMBEDDING_SIZE], -1.0, 1.0),
                name="WordEmbeddingsMatrixTarget"
            )

            self._word_embeddings_context = tf.Variable(
                tf.random_uniform([total_phrases, EMBEDDING_SIZE], -1.0, 1.0),
                name="WordEmbeddingsMatriContext"
            )


            target_word_id_iter = target_word_id_dataset.to_tf_iterator()
            context_word_id_iter = context_word_id_dataset.to_tf_iterator()
            target_word_id_batch = target_word_id_iter.get_next()
            context_word_id_batch = context_word_id_iter.get_next()


            i = tf.reshape(tf.tile(target_word_id_batch, [BATCH_SIZE]), shape=[-1, BATCH_SIZE])
            j = tf.transpose(tf.reshape(tf.tile(context_word_id_batch, [BATCH_SIZE]), shape=[-1,BATCH_SIZE]))
            ij = tf.concat([tf.expand_dims(i, axis=-1), tf.expand_dims(j, axis=-1)], axis=-1)
            x_ij = tf.gather_nd(self._cooccurance_matrix, ij)
            log_x_ij = tf.log(x_ij+1)


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

            xmax = 100.0
            alpha = 0.75
            loss_weight = tf.minimum((x_ij / xmax) ** alpha, 1.0)
            # bias_target = tf.Print(bias_target, [tf.reduce_sum(bias_target)], "BT")

            print(self._word_embeddings_context.shape)
            print(self._word_embeddings_target.shape)

            self._loss = tf.reduce_sum(loss_weight * ((embedding_dot_product + bias_target + bias_context - log_x_ij) ** 2))
            self._word_embeddings = self._word_embeddings_target + self._word_embeddings_context
            print(self._word_embeddings.shape)

            self._optimizer = WitchcraftGradientDescentOptimizer(0.005).to_tf_optimizer().minimize(self._loss)
            self._summary = tf.summary.scalar("loss", self._loss)
            self._writer = tf.summary.FileWriter('./logs/' + "glove", self._session.graph)

            self._session.run(tf.global_variables_initializer(), feed_dict={self._cooccurance_matrix_placeholder: self._cooccurance_matrix_arr})
            #self._session.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()

        print ("Done building graph")

    def train(self, global_step: int) -> None:
        with self._graph.as_default():
            _, calc_summary, loss = self._session.run([self._optimizer, self._summary, self._loss])

            self._writer.add_summary(calc_summary, global_step=global_step)

            if global_step % 100 == 0:
                print("LAST LOSS: " + str(loss))
                self._saver.save(self._session, './logs/' + "glove" + '.ckpt', global_step)

    def save_embeddings(self, filename: str) -> None:
        print("Starting to save...")
        with open(filename, 'wb') as fout:
            embedding_vectors = self._session.run(self._word_embeddings)
            print(embedding_vectors[0])

            for phrase_id in range(len(self._phrase_counts)):
                (phrase, count) = self._phrase_counts[phrase_id]
                embedding_vector = list(embedding_vectors[phrase_id])
                embedding_proto = PhraseEmbedding(phrase, count, embedding_vector).to_protobuf()
                protobuf_to_filestream(fout, embedding_proto.SerializeToString())

        print("Done saving")