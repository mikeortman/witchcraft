from sys import argv
from multiprocessing import Pool
from typing import List

from witchcraft.util.protobuf import protobufs_from_filestream
from projects.naughty.protos.naughty_pb2 import UrbanDictionaryDefinition as UrbanDictionaryDefinitionProto
from projects.naughty.definition import UrbanDictionaryDefinition


def read_file(filename: str) -> List[str]:
    definitions: List[str] = []
    with open(filename, 'rb') as fin:
        for pb_str in protobufs_from_filestream(fin):
            pb = UrbanDictionaryDefinitionProto.FromString(pb_str)
            definition = UrbanDictionaryDefinition.from_protobuf(pb)
            definitions += [str(definition)]

    return definitions


p = Pool(8)
for resultset in p.map(read_file, argv[1:]):
    print(resultset)
