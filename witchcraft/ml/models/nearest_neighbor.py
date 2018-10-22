from typing import List
from witchcraft.ml.datatypes import Embedding
import tensorflow as tf


class NearestNeighborResult:
    def __init__(self, embedding: Embedding, strength: float) -> None:
        self._embedding = embedding
        self._strength = strength

    def get_embedding(self) -> Embedding:
        return self._embedding

    def get_strength(self) -> float:
        return self._strength


class NearestNeighborModel:
    def __init__(self, embeddings: List[Embedding]) -> None:
        self._embeddings = embeddings
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)

        with self._graph.as_default():
            embedding_matrix = tf.constant([e.get_vector() for e in self._embeddings], dtype=tf.float32)
            embedding_normal = tf.sqrt(tf.reduce_sum(tf.square(embedding_matrix), axis=-1))

            embedding_size = embedding_matrix.shape[1]
            self._search_embeddings = tf.placeholder(tf.int32, shape=[None, embedding_size])
            self._result_count = tf.placeholder(tf.int32, shape=[])
            search_embeddings_normal = tf.sqrt(tf.reduce_sum(tf.square(self._search_embeddings), axis=-1))

            search_dot_product = tf.reduce_sum(tf.expand_dims(self._search_embeddings, axis=1) * embedding_matrix, axis=-1)
            cosine_sim = search_dot_product / embedding_normal / tf.expand_dims(search_embeddings_normal, axis=-1)
            self._top_ten_vals, self._top_ten_indicies = tf.nn.top_k(cosine_sim, k=self._result_count, sorted=True)

    def lookup_nearby(self, embedding: Embedding, max_results: int = 10) -> List[NearestNeighborResult]:
        strengths, indicies = self._session.run([self._top_ten_vals, self._top_ten_indicies], feed_dict={
            self._search_embeddings: [embedding.get_vector()],
            self._result_count: max_results
        })

        strengths = strengths[0]
        indicies = indicies[0]

        results = []
        for i in range(len(indicies)):
            current_index = indicies[i]
            current_strength = strengths[i]
            results += [NearestNeighborResult(self._embeddings[current_index], current_strength)]

        return results






