import sys
import json
from multiprocessing import Pool
from tokenembeddings import TokenEmbeddings, TokenEmbeddingTrainingHyperparameters
import logging
from threading import Thread

logging.basicConfig(format='%(name)s [%(levelname)s] %(message)s', level=logging.INFO)
log = logging.getLogger("Trainer")



class Trainer:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def extractTokenArrayFromDefinition(self, tokens):
        def isValidPos(token):
            return token["pos"] not in ['SYM', 'PUNCT', 'SPACE', 'NUM'] and len(token["txt"]) > 1 and token["txt"][
                0] != "'"

        def getActualWord(token):
            # if(token["lem"] == "-PRON-"):
            return token["txt"]

            # return token["lem"]

        return [getActualWord(r) for r in tokens if isValidPos(r)]

    def extractTokensInDefinitions(self):
        for definition in self.definitions:
            for entry in definition["definitions"]:
                yield self.extractTokenArrayFromDefinition(entry["tokens"])

    def run(self):
        log.info("Reading from file...")

        self.definitions = []
        with open("parsed_definitions.json", "r") as ins:
            for line in ins:
                self.definitions += [json.loads(line)]

        log.info("Starting trainer with scope: " + hyperparams.getScopeName())
        embeddings = TokenEmbeddings.build(self.extractTokensInDefinitions, hyperparameters=self.hyperparameters)
        i = 0

        while True:
            embeddings.train(i)
            i += 1

            if i % 1000 == 0:
                log.info(self.hyperparameters.getScopeName() + " trained steps: " + str(i))



def bootstrapTrainerProcess(hyperparameters):
    log.info("TRAINER: " + str(hyperparameters.getScopeName()))
    trainer = Trainer(hyperparameters)
    trainer.run()




embeddings = None # TokenEmbeddings.loadIfExists()
if embeddings is None:

    windowSizes = [2,3]
    learningRates = [0.5,1.0,1.5]
    negSampleCounts = [32]
    batchSizes = [256]
    optimizers = ["adagrad"]

    trainers = []
    for optimizer in optimizers:
        for learningRate in learningRates:
            for negSampleCount in negSampleCounts:
                for windowSize in windowSizes:
                    for batchSize in batchSizes:
                        name = "w" + str(windowSize) + "_b" + str(batchSize) + "_ns" + str(negSampleCount) + "_o" + optimizer + "_l" + str(learningRate)
                        log.info("Creating " + name)
                        hyperparams = TokenEmbeddingTrainingHyperparameters()
                        hyperparams.setScopeName(name)
                        hyperparams.setSkipgramWindowSize(windowSize)
                        hyperparams.setMinimumTokenCount(25)
                        hyperparams.setMaxVocabSize(20000)
                        hyperparams.setLearningRate(learningRate)
                        hyperparams.setBatchSize(batchSize)
                        hyperparams.setOptimizer(optimizer)
                        hyperparams.setMomentum(0.9)
                        trainers += [hyperparams]


    with Pool(20) as p:
        results = p.map(bootstrapTrainerProcess, trainers)

    for trainer in trainers:
        trainer.start()

    log.info("All Threads started... waiting for joins")
    for trainer in trainers:
        trainer.join()

else:
    log.info("Tokens loaded from disk")

log.info ("Done loading!")

