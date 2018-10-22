# @generated by generate_proto_mypy_stubs.py.  Do not edit!
from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from witchcraft.nlp.protos.nlpdatatypes_pb2 import (
    Document as witchcraft___nlp___protos___nlpdatatypes_pb2___Document,
)


class UrbanDictionaryDefinition(google___protobuf___message___Message):
    upvotes = ... # type: int
    downvotes = ... # type: int

    @property
    def word(self) -> witchcraft___nlp___protos___nlpdatatypes_pb2___Document: ...

    @property
    def definition(self) -> witchcraft___nlp___protos___nlpdatatypes_pb2___Document: ...

    def __init__(self,
        word : witchcraft___nlp___protos___nlpdatatypes_pb2___Document,
        definition : witchcraft___nlp___protos___nlpdatatypes_pb2___Document,
        upvotes : int,
        downvotes : int,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> UrbanDictionaryDefinition: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
