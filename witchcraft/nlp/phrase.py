from typing import Generator, List
from witchcraft.nlp.word import Word
from witchcraft.nlp.protos.nlpdatatypes_pb2 import Phrase as PhraseProto

class Phrase:
    def __init__(self):
        self._words: List[Word] = []

    def get_word_generator(self) -> Generator[Word, None, None]:
        for word in self._words:
            yield word

    def to_string(self) -> str:
        # TODO: Make Faster
        return ''.join(
            [w.get_word_string() + w.get_whitespace_postfix() for w in self.get_word_generator()]
        )

    def to_protobuf(self) -> PhraseProto:
        return PhraseProto(
            words=[w.to_protobuf() for w in self.get_word_generator()]
        )
