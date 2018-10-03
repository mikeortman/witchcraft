from projects.naughty.definition import UrbanDictionaryDefinition
from google.protobuf.json_format import MessageToJson

import sys

for definition in UrbanDictionaryDefinition.generate_from_csv(iter(sys.stdin.readline, '')):
    print(MessageToJson(definition.to_protobuf()))