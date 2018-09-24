from typing import Generator, List
from witchcraft.nlp.word import Word
from witchcraft.nlp.phrase import Phrase
from witchcraft.nlp.protos.nlpdatatypes_pb2 import Sentence as SentenceProto

class Sentence:
    def __init__(self):
        self._phrases: List[Phrase] = []

    def get_word_generator(self) -> Generator[Word, None, None]:
        for phrase in self.get_phrase_generator():
            for word in phrase.get_word_generator():
                yield word

    def get_phrase_generator(self) -> Generator[Phrase, None, None]:
        for phrase in self._phrases:
            yield phrase

    def to_string(self) -> str:
        return ''.join([p.to_string() for p in self.get_phrase_generator()])

    def to_protobuf(self) -> SentenceProto:
        return SentenceProto(
            phrases=[p.to_protobuf() for p in self.get_phrase_generator()]
        )
