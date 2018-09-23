from typing import Generator
from nlp.sentence import Sentence
from nlp.protos.nlpdatatypes_pb2 import SentenceSequence as SentenceSequenceProto

class SentenceSequence:
    def __init__(self, from_protobuf: SentenceSequenceProto):
        pass

    def getSentencesGenerator(self) -> Generator[Sentence]:
        pass

    @staticmethod
    def parseString(input_string : str) -> Sentence:
        return SentenceSequence