from typing import Optional

import tensorflow as tf

from witchcraft.ml.optimizers import WitchcraftAdamOptimizer, Optimizer
from witchcraft.ml.datasets import WitchcraftDataset
from witchcraft.nlp.datatypes import Corpus
from witchcraft.ml.datatypes import PhraseEmbedding, PhraseEmbeddingNgram
from witchcraft.util.protobuf import protobuf_to_filestream
from witchcraft.util.listutil import skipgramify

class FastTextHyperparameters:
    def __init__(self) -> None:
        self._embedding_size: int = 300
        self._batch_size: int = 512
        self._window_size: int = 4
        self._subsample_threshold: Optional[float] = None
        self._optimizer: Optimizer = WitchcraftAdamOptimizer(learning_rate=0.001)
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

TOTAL_NGRAM_EMBEDDINGS = 1 # +1 for nil (0)
MAX_VOCAB_SIZE = 50000
MIN_WORD_APPEARANCE = 5
MAX_PHRASE_LEN = 20
NGRAM_LENGTH = 3


class FastTextVocab:
    def __init__(self, corpus: Corpus) -> None:
        allowed_chars = 'abcdefghijklmnopqrstuvwxyz'
        self.ngram_id_lookup = {}

        i = 1  # i = 0 is a nil padding character
        for x in allowed_chars:
            for y in allowed_chars:
                self.ngram_id_lookup["<" + x + y] = i
                i += 1

                self.ngram_id_lookup[x + y + ">"] = i
                i += 1

                for z in allowed_chars:
                    self.ngram_id_lookup[x + y + z] = i
                    i += 1

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
        self._vocab_counts = self._vocab_counts[:MAX_VOCAB_SIZE - 1]

        done_pruning = False
        while not done_pruning:
            print("Pruning...")
            self._ngram_counts = {}
            for (phrase, _) in self._vocab_counts:
                for ngram in self.create_phrase_ngrams(phrase):
                    if ngram not in self._ngram_counts:
                        self._ngram_counts[ngram] = 0

                    self._ngram_counts[ngram] += 1

            self._ngram_counts = [(w, c) for w, c in self._ngram_counts.items() if c >= 5]
            print("Total passing ngrams: " + str(len(self._ngram_counts)))

            ngram_mapping = {w: True for (w, _) in self._ngram_counts}
            pruned_vocab_counts = []
            for (w, c) in self._vocab_counts:
                should_count = True
                for ngram in self.create_phrase_ngrams(w):
                    if ngram not in ngram_mapping:
                        should_count = False
                        break

                if should_count:
                    pruned_vocab_counts += [(w,c)]

            print("Pruned down to " + str(len(pruned_vocab_counts)) + " from " + str(len(self._vocab_counts)))
            if len(pruned_vocab_counts) == len(self._vocab_counts):
                done_pruning = True
            else:
                self._vocab_counts = pruned_vocab_counts

        print ("Done pruning")
        self._ngram_counts = sorted(self._ngram_counts, key=lambda a: -a[1]) #Needed only for debugging below


        i = 1
        self._vocab_to_id = {"[NOP]": 0}
        for (w, c) in self._vocab_counts:
            self._vocab_to_id[w] = i
            i += 1

        i = 1
        self._ngram_to_id = {"[NOP]": 0}
        self._id_to_ngram = ["[NOP]"]
        for (ngram, _) in self._ngram_counts:
            self._ngram_to_id[ngram] = i # 0 is NOP
            self._id_to_ngram += [ngram]
            i += 1

    def __iter__(self):
        return iter(self._vocab_counts)

    def get_ngrams(self):
        return iter(self._id_to_ngram)

    def phrase_to_id(self, phrase):
        return self._vocab_to_id[phrase] if phrase in self._vocab_to_id else None

    def get_vocab_size(self):
        return len(self._vocab_to_id)

    def get_ngrams_size(self):
        return len(self._ngram_to_id)

    def create_phrase_ngrams(self, phrase):
        phrase = "<" + phrase + ">"
        ngrams = []
        #chunk into ngrams.
        for i in range(len(phrase) - NGRAM_LENGTH + 1):
            ngrams += [phrase[i:i+NGRAM_LENGTH]]

        return ngrams

    def create_phrase_ngrams_by_id(self, phrase, max_ngrams):
        current_phrase_ngrams = self.create_phrase_ngrams(phrase)
        current_phrase_ngram_ids = [self._ngram_to_id[ngram] for ngram in current_phrase_ngrams if ngram in self._ngram_to_id]
        if len(current_phrase_ngrams) != len(current_phrase_ngram_ids):
            return None

        # #Concat/Expand to match correct shape
        current_phrase_ngram_ids = current_phrase_ngram_ids[:max_ngrams]
        current_phrase_ngram_ids += [0] * (max_ngrams - len(current_phrase_ngrams))
        return current_phrase_ngram_ids

    def save_ngrams_tsv(self):
        with open("ngrams.tsv", "w") as f:
            for ngram in self._id_to_ngram:
                f.write(ngram + "\n")


class FastTextVocabNgramDataset(WitchcraftDataset):
    def __init__(self, vocab: FastTextVocab) -> None:
        def gen():
            for (phrase, phrase_count) in vocab:
                ngram_arr = vocab.create_phrase_ngrams_by_id(phrase, MAX_PHRASE_LEN)
                if ngram_arr is None:
                    continue

                skipgrams = skipgramify(ngram_arr, 10)
                if skipgrams is not None:
                    # print(skipgrams)
                    for skipgram in skipgrams:
                        yield skipgram

        super(FastTextVocabNgramDataset, self).__init__(tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32, tf.int32)))


class FastTextVocabCorpusDataset(WitchcraftDataset):
    def __init__(self, corpus: Corpus, vocab: FastTextVocab) -> None:
        def gen():
            for doc in corpus:
                for sentence in doc:
                    phrases = [p for p in sentence if vocab.phrase_to_id(p.to_phrase_normalized()) is not None]
                    if len(phrases) <= 1:
                        continue

                    phrases = [vocab.phrase_to_id(p.to_phrase_normalized()) for p in phrases]
                    phrases = phrases[:MAX_PHRASE_LEN]
                    phrases += [0] * (MAX_PHRASE_LEN - len(phrases))

                    for (target, context, distance) in skipgramify(phrases, 10):
                        if target == 0 or context == 0 or distance == 0:
                            continue

                        yield (target, context, distance)

        super(FastTextVocabCorpusDataset, self).__init__(tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32, tf.int32)))


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
        self._vocab.save_ngrams_tsv()

        with self._graph.as_default():
            with tf.variable_scope("NgramVectorization") as scope_ngram:
                self._ngram_train_dataset = FastTextVocabNgramDataset(self._vocab)
                self._ngram_train_dataset = self._ngram_train_dataset.shuffle(shuffle_buffer=1000000)
                self._ngram_train_dataset = self._ngram_train_dataset.repeat()
                self._ngram_train_dataset = self._ngram_train_dataset.batch(batch_size=self._hyperparameters.get_batch_size())

                (target_ngrams, positive_ngram_context_labels, _) = self._ngram_train_dataset.to_tf_iterator().get_next()
                positive_ngram_context_labels = tf.expand_dims(positive_ngram_context_labels, axis=-1, name="ExpandContextLabelIntoVector")

                embedding_size = self._hyperparameters.get_embedding_size()
                self._ngram_embeddings = tf.Variable(
                    tf.random_uniform([self._vocab.get_ngrams_size(), embedding_size], -1.0, 1.0),
                    name="NGramEmbeddings"
                )

                target_ngram_embeddings = tf.gather(self._ngram_embeddings, target_ngrams, name="TargetNGramEmbeddings")

                ngram_nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self._vocab.get_ngrams_size(), embedding_size],
                        stddev=1.0 / embedding_size ** 0.5
                    ),
                    name="NoiseConstrastiveWeights"
                )

                ngram_nce_bias = tf.Variable(tf.zeros([self._vocab.get_ngrams_size()]), name="NoiseConstrastiveBiases")

                self._nce_ngram_loss_batch = tf.nn.nce_loss(
                        weights=ngram_nce_weights,
                        biases=ngram_nce_bias,
                        labels=positive_ngram_context_labels,
                        inputs=target_ngram_embeddings,
                        num_sampled=64,
                        num_classes=self._vocab.get_ngrams_size(),
                        name="NoiseContrastiveLoss"
                )

                self._nce_ngram_loss = tf.reduce_sum(tf.square(self._nce_ngram_loss_batch))
                self._summary_ngram_loss = tf.summary.scalar("ngram_loss_mean", tf.reduce_sum(self._nce_ngram_loss_batch) / self._hyperparameters.get_batch_size())
                self._optimizer_ngram_loss = hyperparameters.get_optimizer().to_tf_optimizer()
                self._minimize_ngram_loss = self._optimizer_ngram_loss.minimize(self._nce_ngram_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_ngram.name))

            with tf.variable_scope("PhraseVectorization") as scope_phrase:
                self._phrase_train_dataset = FastTextVocabCorpusDataset(self._corpus, self._vocab)
                self._phrase_train_dataset = self._phrase_train_dataset.shuffle(shuffle_buffer=500000)
                self._phrase_train_dataset = self._phrase_train_dataset.repeat()
                self._phrase_train_dataset = self._phrase_train_dataset.batch(batch_size=self._hyperparameters.get_batch_size())

                # self._phrase_train_dataset = self._phrase_train_dataset.prefetch(buffer_size=100000)
                (target_phrases, positive_phrases_context_labels, _) = self._phrase_train_dataset.to_tf_iterator().get_next()

                target_phrases = tf.reshape(target_phrases, [self._hyperparameters.get_batch_size()], name="DefineTargetPhraseVector")
                positive_phrases_context_labels = tf.reshape(positive_phrases_context_labels, [self._hyperparameters.get_batch_size(), 1], name="ExpandContextLabelIntoVector")

                self._phrase_ngram_map_placeholder = tf.placeholder(dtype=tf.int32, shape=[self._vocab.get_vocab_size(), MAX_PHRASE_LEN])
                self._phrase_ngram_map = tf.Variable(self._phrase_ngram_map_placeholder, dtype=tf.int32, trainable=False, name="PhraseToNGramMap")

                target_phrase_ngram_maps = tf.gather(self._phrase_ngram_map, target_phrases)
                target_phrase_ngram_embeddings = tf.gather(self._ngram_embeddings, target_phrase_ngram_maps)

                # print("ES", target_phrase_ngram_embeddings)



                # Phrase lengths by counting non zero ngram ids
                with tf.variable_scope("PhraseMaskGeneration"):
                    phrase_length_mask = tf.sign(target_phrase_ngram_maps)
                    phrase_length_mask = tf.maximum(phrase_length_mask, 0)
                    # phrase_length_mask = tf.Print(phrase_length_mask, [phrase_length_mask], "PM")

                    phrase_length = tf.reduce_sum(phrase_length_mask, axis=-1)
                    # phrase_length = tf.Print(phrase_length, [phrase_length], "PL")
                    # target_phrase_ngram_embeddings /= tf.cast(tf.expand_dims(phrase_length, axis=-1), tf.float32)
                    # target_phrase_ngram_embeddings = target_phrase_ngram_embeddings * tf.expand_dims(tf.cast(phrase_length_mask, tf.float32), axis=-1)
                    # attention_optimized_outputs = tf.reduce_sum(target_phrase_ngram_embeddings, axis=-2)


                lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(embedding_size/2, activation=tf.nn.relu)
                lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(embedding_size/2, activation=tf.nn.relu)
                lstm_initial_state_fw = lstm_cell_fw.zero_state(self._hyperparameters.get_batch_size(), dtype=tf.float32)
                lstm_initial_state_bw = lstm_cell_bw.zero_state(self._hyperparameters.get_batch_size(), dtype=tf.float32)

                _, (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                                                                                                lstm_cell_fw,
                                                                                                lstm_cell_bw,
                                                                                                target_phrase_ngram_embeddings,
                                                                                                sequence_length=phrase_length,
                                                                                                initial_state_fw=lstm_initial_state_fw,
                                                                                                initial_state_bw=lstm_initial_state_bw)

                lstm_final_outputs = tf.concat([final_state_fw.h, final_state_bw.h], axis=-1)

                # print("LSTM_FINAL_OUTPUTS", lstm_final_outputs)
                lstm_final_outputs = tf.Print(lstm_final_outputs, [lstm_final_outputs], "LSTM")
                lstm_overextension = tf.reduce_mean(tf.maximum((lstm_final_outputs - 2)**4 - 1, 0) + 1, axis=-1)
                # lstm_overextension = tf.Print(lstm_overextension, [lstm_overextension], "LSTM_OVEREXTENSION")

                lstm_final_outputs_norm = tf.nn.l2_normalize(lstm_final_outputs, axis=-1)
                # lstm_final_outputs_norm = tf.Print(lstm_final_outputs, [lstm_final_outputs_norm], "LSTM_NORM")



                # Hacky stuff. Add some loss to prevent individual features from converging to the same value.
                lstm_mean, lstm_variance = tf.nn.moments(lstm_final_outputs_norm, [-2])
                # lstm_variance = tf.Print(lstm_variance, [lstm_variance], "LSTM_VARIANCE")
                lstm_stddev = tf.sqrt(lstm_variance)
                # lstm_stddev = tf.Print(lstm_stddev, [lstm_stddev], "LSTM_STDDEV")
                lstm_stddev_capped = tf.minimum(lstm_stddev, 0.5)
                # lstm_stddev_capped = tf.Print(lstm_stddev_capped, [lstm_stddev_capped], "LSTM_STDDEV_CAPPED")

                lstm_stddev_loss = tf.reduce_sum(lstm_stddev_capped, axis=-1)
                lstm_stddev_loss = tf.Print(lstm_stddev_loss, [lstm_stddev_loss], "LSTM_STDLOSS")


                # # MASKED ATTENTION
                # with tf.variable_scope("MaskedAttention"):
                #     target_phrase_ngram_embeddings = target_phrase_ngram_embeddings * tf.expand_dims(tf.cast(phrase_length_mask, tf.float32), axis=-1)
                #     flattened_phrase = tf.reshape(target_phrase_ngram_embeddings, [self._hyperparameters.get_batch_size(), -1])
                #
                #     with tf.variable_scope("AttentionNeuralNetwork"):
                #         attention = tf.layers.dense(flattened_phrase, 300, activation=tf.nn.relu)
                #         attention = tf.layers.dense(attention, 128, activation=tf.nn.relu)
                #         attention = tf.layers.dense(attention, 20, activation=tf.nn.relu)
                #         attention = tf.squeeze(attention)
                #
                #     with tf.variable_scope("AttentionMaskedSoftmax"):
                #         attention = tf.exp(attention)
                #         attention = attention * tf.cast(phrase_length_mask, tf.float32)
                #         attention_denom = tf.reduce_sum(attention, axis=-1, keepdims=True)
                #         attention_denom = tf.maximum(attention_denom, 1e-9)
                #         attention = attention / attention_denom
                #
                #     attention = tf.Print(attention, [attention], "AT")
                #
                #     with tf.variable_scope("AttentionOptimizedEmbeddingGen"):
                #         attention_optimized_outputs = target_phrase_ngram_embeddings * tf.expand_dims(attention, axis=-1)
                #         attention_optimized_outputs = tf.reduce_sum(attention_optimized_outputs, axis=-2)


                # with tf.variable_scope("Contextualization"):
                #     context_aware_output = tf.layers.dense(attention_optimized_outputs, 300, activation=tf.nn.relu)
                #     context_aware_output = tf.layers.dense(context_aware_output, 300,
                #                                            activation=tf.nn.relu,
                #                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                #                                            activity_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

                phrase_nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self._vocab.get_vocab_size(), embedding_size],
                        stddev=1.0 / embedding_size ** 0.5
                    ),
                    name="NoiseConstrastiveWeights"
                )

                phrase_nce_bias = tf.Variable(tf.zeros([self._vocab.get_vocab_size()]), name="NoiseConstrastiveBiases")


                self._nce_phrase_loss_batch = tf.nn.nce_loss(
                    weights=phrase_nce_weights,
                    biases=phrase_nce_bias,
                    labels=positive_phrases_context_labels,
                    inputs=lstm_final_outputs,
                    num_sampled=64,
                    num_classes=self._vocab.get_vocab_size(),
                    name="NoiseContrastiveLoss"
                )

                self._nce_phrase_loss = tf.reduce_sum(self._nce_phrase_loss_batch * lstm_overextension)  #50 * lstm_stddev_loss
                self._summary_phrase_loss = tf.summary.scalar("phrase_loss_mean", tf.reduce_sum(self._nce_phrase_loss_batch) / self._hyperparameters.get_batch_size())
                self._summary_phrase_lstm_stddev = tf.summary.scalar("lstm_stddev_loss", lstm_stddev_loss)
                self._summary_phrase_overextension = tf.summary.scalar("lstm_overextension", tf.reduce_sum(lstm_overextension))
                self._summary_phrase = tf.summary.merge([self._summary_phrase_loss, self._summary_phrase_lstm_stddev, self._summary_phrase_overextension])

                self._optimizer_phrase_loss = hyperparameters.get_optimizer().to_tf_optimizer()
                self._minimize_phrase_loss = self._optimizer_phrase_loss.minimize(self._nce_phrase_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_phrase.name))

            self._optimizer_all_loss = hyperparameters.get_optimizer().to_tf_optimizer()
            self._minimize_all_loss = self._optimizer_all_loss.minimize(self._nce_phrase_loss)


                # trainable_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_phrase.name)
                # grads = tf.gradients(self._nce_phrase_loss, trainable_vars)
                # grads, _ = tf.clip_by_global_norm(grads, 0.01)  # gradient clipping
                # grads_and_vars = list(zip(grads, trainable_vars))
                # self._minimize_phrase_loss = self._optimizer_phrase_loss.apply_gradients(grads_and_vars)

            # self._summary = tf.summary.merge([ self._summary_ngram_loss, self._summary_phrase_loss])
            self._writer = tf.summary.FileWriter('./logs/' + self._hyperparameters.get_name(), self._session.graph)

            phrase_ngram_map_arr = []
            phrase_ngram_map_arr += [[0] * MAX_PHRASE_LEN]

            for (phrase, _) in self._vocab:
                phrase_ngram_map_arr += [self._vocab.create_phrase_ngrams_by_id(phrase, MAX_PHRASE_LEN)]

            # print(phrase_ngram_map_arr)

            self._session.run(tf.global_variables_initializer(), feed_dict={self._phrase_ngram_map_placeholder: phrase_ngram_map_arr})

            self._saver = tf.train.Saver()

    def train(self, global_step: int) -> None:
        with self._graph.as_default():
            if global_step <= 25000:
                _, calc_summary = self._session.run([self._minimize_ngram_loss, self._summary_ngram_loss])
                self._writer.add_summary(calc_summary, global_step=global_step)
            elif global_step <= 50000:
                _, ngram_summary, phrase_summary= self._session.run([self._minimize_ngram_loss, self._summary_ngram_loss, self._summary_phrase])
                self._writer.add_summary(ngram_summary, global_step=global_step)
                self._writer.add_summary(phrase_summary, global_step=global_step)
            else:
                _, ngram_summary, phrase_summary= self._session.run([self._minimize_all_loss, self._summary_ngram_loss, self._summary_phrase])
                self._writer.add_summary(ngram_summary, global_step=global_step)
                self._writer.add_summary(phrase_summary, global_step=global_step)

            if global_step % 1000 == 0:
                self._saver.save(self._session, './logs/' + self._hyperparameters.get_name() + '.ckpt', global_step)

    def save_embeddings(self, filename: str) -> None:
        print("Starting to save...")
        with open(filename + ".ngrams.proto", 'wb') as fout:
            ngrams = list(self._vocab.get_ngrams())
            embeddings = self._session.run(self._ngram_embeddings)

            for i in range(len(ngrams)):
                ngram = ngrams[i]
                embedding = embeddings[i]

                # Cheating... reusing phrase embedding for ngram embedding
                embedding_proto = PhraseEmbedding(ngram, 0, embedding).to_protobuf()
                protobuf_to_filestream(fout, embedding_proto.SerializeToString())

        print("Done saving")


