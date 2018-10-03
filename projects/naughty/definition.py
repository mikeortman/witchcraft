from typing import List, Iterable, Optional, Generator
from string import punctuation
import re
import csv

from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from witchcraft.nlp.parse import Parser
from witchcraft.nlp.datatypes import SentenceSequence


class UrbanDictionaryDefinition:
    def __init__(self, csv_row: List[str]) -> None:
        self._upvotes: int = 0
        self._downvotes: int = 0
        self._total_votes: int = 0
        self._effective_votes: int = 0
        self._word: Optional[str] = None
        self._definition: Optional[str] = None
        self._validated: Optional[bool] = None

        if len(csv_row) < 6:
            return

        self._word = csv_row[1].strip().strip(punctuation)
        self._definition = csv_row[5].strip()

        try:
            self._upvotes = int(csv_row[2])
            self._downvotes = int(csv_row[3])
            self._total_votes = self._upvotes + self._downvotes
            self._effective_votes = self._upvotes - self._downvotes
        except ValueError:
            pass  # it was a string, not an int. Sometimes the data is bad

        if self.is_valid():
            self._word = self.clean_definition_string(self._word)
            definition_lines = self._definition.split(';;')
            definition_lines = [self.clean_definition_string(d) for d in definition_lines]
            self._definition = '\n'.join([d for d in definition_lines if d != ""])

            self._word_parsed: SentenceSequence = Parser.parse_string_to_sentence_sequence(self._word)
            self._definition_parsed: SentenceSequence = Parser.parse_string_to_sentence_sequence(self._definition)

    def is_valid(self) -> bool:
        if self._validated is not None:
            return self._validated

        if self._word is None or self._definition is None:
            self._validated = False
            return self._validated

        # Community doesn't approve
        if self._total_votes < 50 or self._effective_votes < 40:
            self._validated = False
            return self._validated

        # Word is too long or too short
        if len(self._word) < 3 or len(self._word) > 32:
            self._validated = False
            return self._validated

        # Something isn't ASCII...
        if not all(ord(c) < 128 for c in self._word):
            self._validated = False
            return self._validated

        if not all(ord(c) < 128 for c in self._definition):
            self._validated = False
            return self._validated

        # The definition contains HTML. Instead of wasting compute cycles removing them,
        # lets just consider this definition invalid
        if '<i>' in self._definition or '<b>' in self._definition or '<strong>' in self._definition:
            self._validated = False
            return self._validated

        # The word shouldn't be in its own definition.
        if self._word in self._definition:
            self._validated = False
            return self._validated

        self._validated = True
        return self._validated

    @staticmethod
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

            str_in = re.sub(r"^(?:#?[0-9a-z]{1,2} *(?:->|=>|-|\.|\)|,)+|\-+) *", "", str_in)  # Remove list ordinals.
            str_in = re.sub(r"[\\[\]]", "", str_in)  # Remove brackets.
            str_in = re.sub(r"\\*", "", str_in)  # Remove asteriks
            str_in = re.sub(r"\\~", "", str_in)  # Remove tildes
            str_in = re.sub(r"[-,.!? ]+$", "", str_in)  # Remove end of line punctuation.

            # Remove pos markers
            str_in = re.sub(
                r"^(?:[(\[]?(?:noun|n|verb|v|adverb|adv|av|adjective|adj|aj)[\.,;]*[)\].]*[^a-z\n])+",
                "",
                str_in
            )

        return str_in.strip()

    def to_protobuf(self):
        return UrbanDictionaryDefinitionProto(
            word=self._word_parsed.to_protobuf(),
            definition=self._definition_parsed.to_protobuf(),
            upvotes=self._upvotes,
            downvotes=self._downvotes
        )

    @classmethod
    def generate_from_csv(cls, input_csv_iter: Iterable) -> Generator['UrbanDictionaryDefinition', None, None]:
        for row in csv.reader(input_csv_iter):
            definition = UrbanDictionaryDefinition(row)
            if definition.is_valid():
                yield definition
