from typing import List, Optional
from witchcraft.ml.protos.mldatatypes_pb2 import Embedding as EmbeddingProto
from witchcraft.ml.protos.mldatatypes_pb2 import PhraseEmbedding as PhraseEmbeddingProto
from witchcraft.ml.protos.mldatatypes_pb2 import PhraseEmbeddingNgram as PhraseEmbeddingNgramProto


class Embedding:
    def __init__(self, vector: List[float]) -> None:
        self._vector = list(vector)

    def get_vector(self) -> List[float]:
        return self._vector


class PhraseEmbedding(Embedding):
    def __init__(self, phrase: str, count: int, vector: List[float], ngrams: Optional[List['PhraseEmbeddingNgram']] = None) -> None:
        Embedding.__init__(self, vector=vector)
        self._word = phrase
        self._count = count
        self._ngrams = [] if ngrams is None else ngrams

    def get_phrase(self) -> str:
        return self._word

    def get_count(self) -> int:
        return self._count

    def get_ngrams(self) -> List['PhraseEmbeddingNgram']:
        return self._ngrams

    def to_protobuf(self) -> PhraseEmbeddingProto:
        return PhraseEmbeddingProto(
            phrase=self.get_phrase(),
            count=self.get_count(),
            embedding=EmbeddingProto(embedding_vector=self.get_vector()),
            ngrams=[n.to_protobuf() for n in self._ngrams]
        )

    @classmethod
    def from_protobuf(cls, phrase_embedding_proto: PhraseEmbeddingProto) -> 'PhraseEmbedding':
        return PhraseEmbedding(
            phrase=phrase_embedding_proto.phrase,
            count=phrase_embedding_proto.count,
            vector=phrase_embedding_proto.embedding.embedding_vector,
            ngrams=[PhraseEmbeddingNgram.from_protobuf(a) for a in phrase_embedding_proto.ngrams]
        )

class PhraseEmbeddingNgram:
    def __init__(self, ngram: str, attention: float) -> None:
        self._ngram = ngram
        self._attention = attention

    def get_ngram(self) -> str:
        return self._ngram

    def get_attention(self) -> float:
        return self._attention

    def to_protobuf(self) -> PhraseEmbeddingNgramProto:
        return PhraseEmbeddingNgramProto(
            ngram=self.get_ngram(),
            attention=self.get_attention()
        )

    @classmethod
    def from_protobuf(cls, ngram_proto: PhraseEmbeddingNgramProto) -> 'PhraseEmbeddingNgram':
        return PhraseEmbeddingNgram(
            ngram=ngram_proto.ngram,
            attention=ngram_proto.attention
        )
