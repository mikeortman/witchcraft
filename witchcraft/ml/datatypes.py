from typing import List
from witchcraft.ml.protos.mldatatypes_pb2 import Embedding as EmbeddingProto
from witchcraft.ml.protos.mldatatypes_pb2 import PhraseEmbedding as PhraseEmbeddingProto


class Embedding:
    def __init__(self, vector: List[float]) -> None:
        self._vector = list(vector)

    def get_vector(self) -> List[float]:
        return self._vector


class PhraseEmbedding(Embedding):
    def __init__(self, phrase: str, count: int, vector: List[float]) -> None:
        Embedding.__init__(self, vector=vector)
        self._word = phrase
        self._count = count

    def get_phrase(self) -> str:
        return self._word

    def get_count(self) -> int:
        return self._count

    def to_protobuf(self) -> PhraseEmbeddingProto:
        return PhraseEmbeddingProto(
            phrase=self.get_phrase(),
            count=self.get_count(),
            embedding=EmbeddingProto(embedding_vector=self.get_vector())
        )

    @classmethod
    def from_protobuf(cls, phrase_embedding_proto: PhraseEmbeddingProto) -> 'PhraseEmbedding':
        return PhraseEmbedding(
            phrase=phrase_embedding_proto.phrase,
            count=phrase_embedding_proto.count,
            vector=phrase_embedding_proto.embedding.embedding_vector
        )
