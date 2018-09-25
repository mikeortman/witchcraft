# from typing import Optional
#
# class Word2VecHyperparameters:
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
# class Word2Vec:
#     # Classic word2vec model for use with witchcraft
#     def __init__(self, hyperparameters:Optional[Word2VecHyperparameters] = None):
#         if hyperparameters is None:
#             hyperparameters = Word2VecHyperparameters()
#
#         self.hyperparameters = hyperparameters
