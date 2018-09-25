# @generated by generate_proto_mypy_stubs.py.  Do not edit!
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    Optional as typing___Optional,
    Text as typing___Text,
)


class PartOfSpeech(google___protobuf___message___Message):
    pos = ... # type: typing___Text

    def __init__(self,
        pos : typing___Text,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> PartOfSpeech: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class Word(google___protobuf___message___Message):
    word = ... # type: typing___Text
    lemma = ... # type: typing___Text
    isStopWord = ... # type: bool
    shape = ... # type: typing___Text
    postWhitespace = ... # type: typing___Text
    isAlphaWord = ... # type: bool

    @property
    def partOfSpeech(self) -> PartOfSpeech: ...

    def __init__(self,
        partOfSpeech : PartOfSpeech,
        word : typing___Text,
        lemma : typing___Text,
        isStopWord : bool,
        shape : typing___Text,
        postWhitespace : typing___Text,
        isAlphaWord : bool,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Word: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class Phrase(google___protobuf___message___Message):

    @property
    def words(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Word]: ...

    def __init__(self,
        words : typing___Optional[typing___Iterable[Word]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Phrase: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class Sentence(google___protobuf___message___Message):

    @property
    def phrases(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Phrase]: ...

    def __init__(self,
        phrases : typing___Optional[typing___Iterable[Phrase]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Sentence: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class SentenceSequence(google___protobuf___message___Message):

    @property
    def sentences(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Sentence]: ...

    def __init__(self,
        sentences : typing___Optional[typing___Iterable[Sentence]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> SentenceSequence: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...