from typing import Generator, List
from witchcraft.nlp.sentence import Sentence
from witchcraft.nlp.protos.nlpdatatypes_pb2 import SentenceSequence as SentenceSequenceProto

class SentenceSequence:
    def __init__(self):
        self._sentences: List[Sentence] = []

    def get_sentence_generator(self) -> Generator[Sentence, None, None]:
        for sentence in self._sentences:
            yield sentence

    def to_protobuf(self) -> SentenceSequenceProto:
        return SentenceSequenceProto(
            sentences=[s.to_protobuf() for s in self.get_sentence_generator()]
        )
