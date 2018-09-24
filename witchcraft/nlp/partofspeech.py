from witchcraft.nlp.protos.nlpdatatypes_pb2 import PartOfSpeech as PartOfSpeechProto
class PartOfSpeech:
    def __init__(self):
        self._pos = ""

    def to_string(self) -> str:
        return self._pos

    def to_protobuf(self) -> PartOfSpeechProto:
        return PartOfSpeechProto(
            pos=self._pos
        )
