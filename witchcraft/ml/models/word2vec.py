from typing import Optional, Generator, Tuple
from witchcraft.ml.optimizers import Optimizer
import tensorflow as tf


class Word2VecHyperparameters:
    def __init__(self) -> None:
        self._embedding_size: int = 300
        self._batch_size: int = 64
        self._negative_sample_count: int = 64
        self._min_token_count: int = 20
        self._max_vocab_size: Optional[int] = None
        self._window_size: int = 5
        self._subsample_threshold: Optional[float] = None
        self._optimizer: Optimizer = Optimizer()
        self._name: Optional[str] = None

    def set_embedding_size(self, embedding_size: int) -> None:
        self._embedding_size = embedding_size

    def get_embedding_size(self) -> int:
        return self._embedding_size

    def set_batch_size(self, batch_size) -> None:
        self._batch_size = batch_size

    def get_batch_size(self) -> int:
        return self._batch_size

    def set_negative_sample_count(self, negative_sample_count: int) -> None:
        self._negative_sample_count = negative_sample_count

    def get_negative_sample_count(self) -> int:
        return self._negative_sample_count

    def get_name(self) -> str:
        return self._name if self._name is not None else "<default>"

    def set_name(self, name) -> None:
        self._name = name

    def set_minimum_token_count(self, min_token_count) -> None:
        self._min_token_count = min_token_count

    def get_minimum_token_count(self) -> Optional[int]:
        return self._min_token_count

    def set_max_vocab_size(self, max_vocab_size) -> None:
        self._max_vocab_size = max_vocab_size

    def get_max_vocab_size(self) -> Optional[int]:
        return self._max_vocab_size

    def set_skipgram_window_size(self, window_size) -> None:
        self._window_size = window_size

    def get_skipgram_window_size(self) -> int:
        return self._window_size

    def set_subsampling_threshold(self, subsample_threshold) -> None:
        self._subsample_threshold = subsample_threshold

    def get_subsampling_threshold(self) -> Optional[float]:
        return self._subsample_threshold

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def get_optimizer(self) -> Optimizer:
        return self._optimizer


class Word2VecModel:
    # Classic word2vec model for use with witchcraft
    def __init__(
        self,
        training_pair_generator: Generator[Tuple[str, str], None, None],
        vocab_size: int,
        hyperparameters: Optional[Word2VecHyperparameters] = None
    ):
        if hyperparameters is None:
            hyperparameters = Word2VecHyperparameters()

        self._hyperparameters: Word2VecHyperparameters = hyperparameters
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

        with self._graph.as_default():
            self._dataset = tf.data.Dataset.from_generator(
                training_pair_generator,
                (tf.int32, tf.int32),
                output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
            )

            self._dataset = self._dataset.shuffle(buffer_size=500000)
            self._dataset = self._dataset.repeat()
            self._dataset = self._dataset.prefetch(buffer_size=1000000)
            self._dataset = self._dataset.make_one_shot_iterator()

            (center_words, target_words) = self._dataset.get_next()
            target_words = tf.expand_dims(target_words, axis=-1)

            embedding_size = self._hyperparameters.get_embedding_size()
            self._word_embed_matrix = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="WordEmbeddingsMatrix"
            )

            center_embeddings = tf.nn.embedding_lookup(self._word_embed_matrix, center_words, name="WordEmbeddings")

            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocab_size, embedding_size],
                    stddev=1.0 / embedding_size ** 0.5
                ),
                name="NoiseConstrastiveWeights"
            )

            nce_bias = tf.Variable(tf.zeros([vocab_size]), name="NoiseConstrastiveBiases")

            self._loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_bias,
                    labels=target_words,
                    inputs=center_embeddings,
                    num_sampled=self._hyperparameters.get_negative_sample_count(),
                    num_classes=vocab_size
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
