from witchcraft.nlp.parse import Parser
from witchcraft.nlp.datatypes import SentenceSequence

def test_singleton_spacy():
    proto_built = Parser.parse_string_to_sentence_sequence("Winner! Winner! We have a winner!").to_protobuf()
    rebuild_sequence = SentenceSequence.from_protobuf(proto_built)
    proto_rebuilt = rebuild_sequence.to_protobuf()
    assert proto_built == proto_rebuilt
