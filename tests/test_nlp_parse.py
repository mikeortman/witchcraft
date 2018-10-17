from witchcraft.nlp.parse import Parser
from witchcraft.nlp.datatypes import SentenceSequence

def test_singleton_spacy():
    proto_built = Parser.parse_string_to_sentence_sequence("Winner! Winner! We have a winner!").to_protobuf()
    rebuild_sequence = SentenceSequence.from_protobuf(proto_built)
    proto_rebuilt = rebuild_sequence.to_protobuf()
    assert proto_built == proto_rebuilt

def test_phrase_grouping():
    str = "Mike Ortman was here. When Mike Ortman was bored, " \
          "he created witchcraft. Mr. Mike Ortman is getting old! " \
          "Andy is a brother of Mike. Ortman is their last name. " \
          "Andy is a twin of Mike. " \
          "Mr. Mike Ortman is Mike's name. " \
          "And one more: Mr. Mike Ortman"

    parsed = Parser.parse_string_to_sentence_sequence(str)
    for sentence in parsed.get_sentence_generator():
        print("\n" + sentence.__str__())
        for phrase in sentence.get_phrase_generator():
            print("\t" + phrase.__str__())

    print ("Clustered!")
    for sequence in Parser.cluster_phrases(parsed.get_sequence_generator, 3, 2)():
        for sentence in sequence.get_sentence_generator():
            print("\n" + sentence.__str__())
            for phrase in sentence.get_phrase_generator():
                print("\t" + phrase.__str__())

    assert False
