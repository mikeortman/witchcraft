from typing import Generator
from nlp.word import Word
from nlp.protos.nlpdatatypes_pb2 import PartOfSpeech as PartOfSpeedProto

class Phrase:
    def __init(self, from_protobuf: PartOfSpeedProto):
        pass

    def getWordGenerator(self) -> Generator[Word]:
        pass

