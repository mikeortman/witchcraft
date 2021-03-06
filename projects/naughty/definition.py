from typing import List, Iterable, Generator
import re
import csv

from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from witchcraft.nlp.parse import parse_string_to_document
from witchcraft.nlp.datatypes import Document


class UrbanDictionaryDefinition:
    def __init__(self, upvotes: int, downvotes: int, word: Document, definition: Document) -> None:
        self._upvotes: int = upvotes
        self._downvotes: int = downvotes
        self._word: Document = word
        self._definition: Document = definition

    def get_word(self) -> Document:
        return self._word

    def get_definition(self) -> Document:
        return self._definition

    def get_upvotes(self) -> int:
        return self._upvotes

    def get_downvotes(self) -> int:
        return self._downvotes

    def __str__(self):
        return str(self.get_word())

    def to_protobuf(self):
        return UrbanDictionaryDefinitionProto(
            upvotes=self._upvotes,
            downvotes=self._downvotes,
            word=self.get_word().to_protobuf(),
            definition=self.get_definition().to_protobuf()
        )

    def to_array(self) -> List[any]:
        return [
            self.get_upvotes(),
            self.get_downvotes(),
            self.get_word().to_array(),
            self.get_definition().to_array()
        ]

    @classmethod
    def from_protobuf(cls, protobuf: UrbanDictionaryDefinitionProto) -> 'UrbanDictionaryDefinition':
        return UrbanDictionaryDefinition(
            upvotes=protobuf.upvotes,
            downvotes=protobuf.downvotes,
            word=Document.from_protobuf(protobuf.word),
            definition=Document.from_protobuf(protobuf.definition)
        )

    @classmethod
    def from_array(cls, arr: List[any]) -> 'UrbanDictionaryDefinition':
        return UrbanDictionaryDefinition(
            upvotes=arr[0],
            downvotes=arr[1],
            word=Document.from_array(arr[2]),
            definition=Document.from_array(arr[3])
        )


def clean_definition_string(str_in: str) -> str:
    # Remove letters that appear more than twice
    str_in = str_in.strip()
    if len(str_in) > 3:
        new_str: List[str] = [str_in[0], str_in[1], str_in[2]]
        for i in range(3, len(str_in)):
            great_grandparent = str_in[i - 3]
            grandparent = str_in[i - 2]
            parent = str_in[i - 1]
            child = str_in[i]

            if great_grandparent is child and grandparent is child and parent is child:
                continue

            new_str += [child]

        str_in = ''.join(new_str)

        str_in = re.sub(r"^(?:#?[0-9a-z]{1,2} *(?:->|=>|-|\.|\)|,)+|-+) *", "", str_in)  # Remove list ordinals.
        str_in = re.sub(r"[\\[\]]", "", str_in)  # Remove brackets.
        str_in = re.sub(r"\\*", "", str_in)  # Remove asteriks
        str_in = re.sub(r"\\~", "", str_in)  # Remove tildes
        str_in = re.sub(r"[-,.!? ]+$", "", str_in)  # Remove end of line punctuation.

        # Remove pos markers
        str_in = re.sub(
            r"^(?:[(\[]?(?:noun|n|verb|v|adverb|adv|av|adjective|adj|aj)[.,;]*[)\].]*[^a-z\n])+",
            "",
            str_in
        )

    return str_in.strip()


def generate_definitions_from_csv(input_csv_iter: Iterable) -> Generator[UrbanDictionaryDefinition, None, None]:
    for csv_row in csv.reader(input_csv_iter):
        if len(csv_row) < 6:
            continue

        word: str = csv_row[1].strip()
        definition = csv_row[5].strip()

        try:
            upvotes = int(csv_row[2])
            downvotes = int(csv_row[3])
            total_votes = upvotes + downvotes
            effective_votes = upvotes - downvotes
        except ValueError:
            continue  # it was a string, not an int. Sometimes the data is bad

        if word is None or definition is None:
            continue

        # Community doesn't approve
        if total_votes < 50 or effective_votes < 40:
            continue

        # Word is too long or too short
        if len(word) < 3 or len(word) > 32:
            continue

        # Something isn't ASCII...
        if not all(ord(c) < 128 for c in word):
            continue

        if not all(ord(c) < 128 for c in definition):
            continue

        # The definition contains HTML. Instead of wasting compute cycles removing them,
        # lets just consider this definition invalid
        if '<i>' in definition or '<b>' in definition or '<strong>' in definition:
            continue

        # The word shouldn't be in its own definition.
        if word in definition:
            continue

        word = clean_definition_string(word)
        definition_lines: List[str] = definition.split(';;')
        definition_lines = [clean_definition_string(d) for d in definition_lines]
        definition = '\n'.join([d for d in definition_lines if d != ""])

        word_parsed: Document = parse_string_to_document(word)
        definition_parsed: Document = parse_string_to_document(definition)

        yield UrbanDictionaryDefinition(
            upvotes=upvotes,
            downvotes=downvotes,
            word=word_parsed,
            definition=definition_parsed
        )
