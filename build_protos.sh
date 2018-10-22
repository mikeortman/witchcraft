protoc -I. --mypy_out=. --python_out=. \
	projects/naughty/protos/naughty.proto \
	witchcraft/nlp/protos/nlpdatatypes.proto \
	witchcraft/ml/protos/mldatatypes.proto
