from typing import Generator
from nlp.word import Word
from nlp.phrase import Phrase
from nlp.protos.nlpdatatypes_pb2 import Sentence as SentenceProto

class Sentence:
    def __init__(self, from_protobuf: SentenceProto):
        pass

    def getWordGenerator(self) -> Generator[Word]:
        pass

    def getPhraseGenerator(self) -> Generator[Phrase]:
        pass

    def toString(self) -> str:
        pass