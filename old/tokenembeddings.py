# import tensorflow as tf
# import os
# from tensorflow.contrib.tensorboard.plugins import projector
# import logging
#
# class Token:
#     def __init__(self, token):
#         self.token = token
#
#     def __eq__(self, other):
#         if isinstance(other, Token):
#             return self.token == other.token
#         return NotImplemented
#
#     def getTokenAsString(self):
#         return self.token
#
#
# class TokenEmbeddingTrainingHyperparameters:
#     def __init__(self):
#         self.embeddingSize = 300
#         self.batchSize = 64
#         self.learningRate = 1.0
#         self.negativeSampleCount = 64
#         self.minTokenCount = 20
#         self.maxVocabSize = None
#         self.windowSize = 5
#         self.subsampleThreshold = 0.001
#         self.optimizer = "sgd"
#         self.momentum = 0.1
#         self.name = "<default>"
#
#     def setEmbeddingSize(self, size):
#         self.embeddingSize = size
#
#     def getEmbeddingSize(self):
#         return self.embeddingSize
#
#     def setBatchSize(self, size):
#         self.batchSize = size
#
#     def getBatchSize(self):
#         return self.batchSize
#
#     def setLearningRate(self, rate):
#         self.learningRate = rate
#
#     def getLearningRate(self):
#         return self.learningRate
#
#     def setNegativeSampleCount(self, count):
#         self.negativeSampleCount = count
#
#     def getNegativeSampleCount(self):
#         return self.negativeSampleCount
#
#     def getScopeName(self):
#         return self.name
#
#     def setScopeName(self, name):
#         self.name = name
#
#     def setMinimumTokenCount(self, min_count):
#         self.minTokenCount = min_count
#
#     def getMinimumTokenCount(self):
#         return self.minTokenCount
#
#     def setMaxVocabSize(self, max_size):
#         self.maxVocabSize = max_size
#
#     def getMaxVocabSize(self):
#         return self.maxVocabSize
#
#     def setSkipgramWindowSize(self, window_size):
#         self.windowSize = window_size
#
#     def getSkipgramWindowSize(self):
#         return self.windowSize
#
#     def getSubsamplingThreshold(self, threshold):
#         self.subsampleThreshold = threshold
#
#     def getSubsamplingThreshold(self):
#         return self.subsampleThreshold
#
#     def setOptimizer(self, optimizer):
#         self.optimizer = optimizer
#
#     def getOptimizer(self):
#         return self.optimizer
#
#     def setMomentum(self, momentum):
#         self.momentum = momentum
#
#     def getMomentum(self):
#         return self.momentum
#
# class TokenEmbeddingModel:
#     def __init__(self, training_pair_generator, vocab_size, hyperparameters=None):
#         if hyperparameters is None:
#             hyperparameters = TokenEmbeddingTrainingHyperparameters()
#
#         self.graph = tf.Graph()
#         self.session = tf.Session(graph=self.graph)
#         self.hyperparams = hyperparameters
#
#         with self.graph.as_default():
#             dataset = tf.data.Dataset.from_generator(training_pair_generator, (tf.int32, tf.int32), output_shapes=(tf.TensorShape([None]), tf.TensorShape([None])))
#             self.datasetIterator = dataset.shuffle(buffer_size=500000).repeat().prefetch(buffer_size=1000000).make_one_shot_iterator()
#
#             (centerWords, targetWords) = self.datasetIterator.get_next()
#             targetWords = tf.expand_dims(targetWords, axis=-1)
#
#             embeddingSize = self.hyperparams.getEmbeddingSize()
#             self.wordEmbeddingsMatrix = tf.Variable(tf.random_uniform([vocab_size, embeddingSize], -1.0, 1.0), name="WordEmbeddingsMatrix")
#             centerEmbedding = tf.nn.embedding_lookup(self.wordEmbeddingsMatrix, centerWords, name="WordEmbeddings")
#
#             nceWeights = tf.Variable(tf.truncated_normal([vocab_size, embeddingSize], stddev=1.0 / embeddingSize ** 0.5), name="NoiseConstrastiveWeights")
#             nceBias = tf.Variable(tf.zeros([vocab_size]), name="NoiseConstrastiveBiases")
#
#             self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nceWeights,
#                                                  biases=nceBias,
#                                                  labels=targetWords,
#                                                  inputs=centerEmbedding,
#                                                  num_sampled=self.hyperparams.getNegativeSampleCount(),
#                                                  num_classes=vocab_size), name="MeanNCELoss")
#
#             optimizerName = hyperparameters.getOptimizer().strip().lower()
#             if optimizerName == "adagrad":
#                 self.optimizer = tf.train.AdagradOptimizer(self.hyperparams.getLearningRate()).minimize(self.loss)
#             elif optimizerName == "adam":
#                 self.optimizer = tf.train.AdamOptimizer(self.hyperparams.getLearningRate()).minimize(self.loss)
#             elif optimizerName == "momentum":
#                 self.optimizer = tf.train.MomentumOptimizer(self.hyperparams.getLearningRate(), self.hyperparams.getMomentum()).minimize(self.loss)
#             else:
#                 self.optimizer = tf.train.GradientDescentOptimizer(self.hyperparams.getLearningRate()).minimize(self.loss)
#
#
#             self.summary = tf.summary.scalar("loss", self.loss)
#
#
#             self.writer = tf.summary.FileWriter('./logs/' + self.hyperparams.getScopeName(), self.session.graph)
#             self.session.run(tf.global_variables_initializer())
#             self.saver = tf.train.Saver()
#
#     def train(self, global_step):
#         with self.graph.as_default():
#             _ , calc_summary = self.session.run([self.optimizer, self.summary])
#             self.writer.add_summary(calc_summary, global_step=global_step)
#
#             if global_step % 1000 == 0:
#                 self.saver.save(self.session, './logs/' + self.hyperparams.getScopeName() + '.ckpt', global_step)
#
#
# class TokenEmbeddings:
#     def __init__(self, model):
#         self.model = model
#
#     def train(self, global_step):
#         self.model.train(global_step)
#
#     @staticmethod
#     def sentenceGeneratorTokenFilter(sentence_generator, token_to_id):
#         for sentence in sentence_generator():
#             sentence = [token_to_id[t] if t in token_to_id else None for t in sentence]
#             if len([t for t in sentence if t is not None]) < 2:
#                 continue
#
#             yield sentence
#
#     @staticmethod
#     def sentenceGeneratorToSkipgramGenerator(sentence_generator, token_to_id, hyperparameters):
#         print ("started sentence generator!")
#         def skipgramGen():
#             for sentence in TokenEmbeddings.sentenceGeneratorTokenFilter(sentence_generator, token_to_id):
#                 sentenceSize = len(sentence)
#                 windowSize = hyperparameters.getSkipgramWindowSize()
#
#
#                 for i in range(sentenceSize):
#                     currentToken = sentence[i]
#                     if currentToken is None:
#                         continue
#                     for y in range(1, windowSize + 1):
#                         # print("Y" + str(y))
#                         if i - y >= 0 and sentence[i - y] is not None:
#                             yield (currentToken, sentence[i - y])
#
#                     for y in range(1, windowSize + 1):
#                         if i + y < sentenceSize and sentence[i + y] is not None:
#                             yield (currentToken, sentence[i + y])
#
#         batchSource = []
#         batchTarget = []
#         i = 0
#         for source, target in skipgramGen():
#             batchSource += [source]
#             batchTarget += [target]
#             i += 1
#             if i == hyperparameters.getBatchSize():
#                 i = 0
#                 yield (batchSource, batchTarget)
#                 batchSource = []
#                 batchTarget = []
#
#         if i != 0:
#             yield (batchSource, batchTarget)
#
#     @staticmethod
#     def build(sentence_generator, hyperparameters=None):
#         if hyperparameters is None:
#             hyperparameters = TokenEmbeddingTrainingHyperparameters()
#
#         log = logging.getLogger("[TokenEmbeddings] [" + hyperparameters.getScopeName() + "] [build]")
#
#         log.info("Counting tokens...")
#         tokenCounts = {}
#         for sentence in sentence_generator():
#             for token in sentence:
#                 tokenCounts[token] = tokenCounts.get(token, 0) + 1
#
#         log.info("Total tokens before: " + str(len(tokenCounts)) + ". Clipping to  minimum token count requirement...")
#         tokenCounts = [(k,v) for k,v in tokenCounts.items() if v >= hyperparameters.getMinimumTokenCount()]
#
#
#         # Subsampling
#         if hyperparameters.getSubsamplingThreshold() is not None:
#             log.info("Subsampling enabled. Calculating corpus size...")
#             totalTokensInCorpus = 0
#             for token, count in tokenCounts:
#                 totalTokensInCorpus += count
#
#             log.info("Total tokens: " + str(totalTokensInCorpus) + ". Performing subsampling...")
#             tokensToKeep = []
#             for token, count in tokenCounts:
#                 tokenFrequency = count / totalTokensInCorpus
#                 if tokenFrequency >= hyperparameters.getSubsamplingThreshold():
#                     log.info("Removing '" + token + "' during subsampling; frequency: " + str(tokenFrequency * 100) + "%")
#                     continue
#
#                 tokensToKeep += [(token, count)]
#
#             tokenCounts = tokensToKeep
#         else:
#             log.info("Subsampling disabled.")
#
#
#
#
#         log.info("Sorting token counts")
#         tokenCounts = sorted(tokenCounts, key=lambda a: -a[1])
#
#         # log.info("Clipping out: " + str(tokenCounts[:75]))
#
#         log.info("Total unique tokens to sample: " + str(len(tokenCounts)) + "... clipping to max vocab size if applicable.")
#         if hyperparameters.getMaxVocabSize() is not None:
#             tokenCounts = tokenCounts[:hyperparameters.getMaxVocabSize()]
#
#         log.info("Clipped vocab size. Final vocab size: " + str(len(tokenCounts)) + ". Building unique token index")
#
#
#         uniqueTokens = {}
#         for token, count in tokenCounts:
#             uniqueTokens[token] = True
#
#         log.info("Cleaning sentences")
#
#
#         log.info("Building token id map...")
#         tokenToId = {}
#         idToToken = []
#         i = 0
#         for token, count in tokenCounts:
#             if token is None:
#                 continue
#
#             if token not in tokenToId:
#                 tokenToId[token] = i
#                 idToToken += [token]
#                 i += 1
#
#         log.info("Saving token file...")
#         with open("tokens_" + hyperparameters.getScopeName() + ".tsv", "w") as tokenMetadataFile:
#             for token in idToToken:
#                 tokenMetadataFile.write(token + "\n")
#
#         # print ("Building probability distribution for negative sampling")
#         # tokenProbabilityDistribution = np.zeros(len(tokenToId))
#         # tokenChoiceIndicies = np.arange(len(tokenToId))
#         # for sentence in sentences:
#         #     for token in sentence:
#         #         tokenProbabilityDistribution[tokenToId[token]] += 1
#         #
#         # tokenProbabilityDistribution = np.power(tokenProbabilityDistribution, 3/4)
#         # tokenProbabilityDistribution /= np.sum(tokenProbabilityDistribution)
#         #
#         # print(tokenProbabilityDistribution)
#
#         # Setup the embeddings
#
#         vocabularySize = len(tokenToId)
#         log.info("Final vocab size: " + str(vocabularySize))
#
#
#         def skipgramGenerator():
#             return TokenEmbeddings.sentenceGeneratorToSkipgramGenerator(sentence_generator, tokenToId, hyperparameters)
#
#         model = TokenEmbeddingModel(skipgramGenerator, vocabularySize, hyperparameters=hyperparameters)
#
#         config = projector.ProjectorConfig()
#         embedding_config = config.embeddings.add()
#         embedding_config.tensor_name = model.wordEmbeddingsMatrix.name
#         embedding_config.metadata_path = os.path.join(os.getcwd(), "tokens_" + hyperparameters.getScopeName() + ".tsv")
#         projector.visualize_embeddings(model.writer, config)
#
#         return model
#
#
#     # def save(self):
#     #     with open(TOKEN_MAP_JSON_FILE, "w") as outputFile:
#     #         outputFile.write(json.dumps(self.tokenIdMap))
#     #
#     #     with open(TOKEN_CONTEXT_MATRIX_FILE, "w") as outputFile:
#     #         np.save(outputFile, self.tokenContextMatrix)
#
#     # @staticmethod
#     # def loadIfExists():
#     #     if not os.path.isfile(TOKEN_CONTEXT_MATRIX_FILE):
#     #         return None
#     #
#     #     if not os.path.isfile(TOKEN_MAP_JSON_FILE):
#     #         return None
#     #
#     #     with open(TOKEN_MAP_JSON_FILE, "r") as inputFile:
#     #         tokenIdMap = json.loads(inputFile.read())
#     #
#     #     tokenContextMatrix = np.load(TOKEN_CONTEXT_MATRIX_FILE)
#     #
#     #     return WordEmbeddings(tokenIdMap, tokenContextMatrix)
