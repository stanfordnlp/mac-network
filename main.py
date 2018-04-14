from __future__ import division
import sys 
import os
import time
import math
import random
try:
    import Queue as queue
except ImportError:
    import queue
import threading
import h5py
import json
import numpy as np
import tensorflow as tf
from termcolor import colored, cprint

from config import config, loadDatasetConfig, parseArgs
from preprocess import Preprocesser, bold, bcolored, writeline, writelist
from model import MACnet

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

############################################# loggers #############################################

# Writes log header to file 
def logInit():
    with open(config.logFile(), "a+") as outFile:
        writeline(outFile, config.expName)
        headers = ["epoch", "trainAcc", "valAcc", "trainLoss", "valLoss"]
        if config.evalTrain:
            headers += ["evalTrainAcc", "evalTrainLoss"]
        if config.extra:
            if config.evalTrain:
                headers += ["thAcc", "thLoss"]
            headers += ["vhAcc", "vhLoss"]
        headers += ["time", "lr"]

        writelist(outFile, headers)
        # lr assumed to be last

# Writes log record to file 
def logRecord(epoch, epochTime, lr, trainRes, evalRes, extraEvalRes):
    with open(config.logFile(), "a+") as outFile:
        record = [epoch, trainRes["acc"], evalRes["val"]["acc"], trainRes["loss"], evalRes["val"]["loss"]]
        if config.evalTrain:
            record += [evalRes["evalTrain"]["acc"], evalRes["evalTrain"]["loss"]]
        if config.extra:
            if config.evalTrain:
                record += [extraEvalRes["evalTrain"]["acc"], extraEvalRes["evalTrain"]["loss"]]
            record += [extraEvalRes["val"]["acc"], extraEvalRes["val"]["loss"]]
        record += [epochTime, lr]

        writelist(outFile, record)

# Gets last logged epoch and learning rate
def lastLoggedEpoch():
    with open(config.logFile(), "r") as inFile:
        lastLine = list(inFile)[-1].split(",") 
    epoch = int(lastLine[0])
    lr = float(lastLine[-1])   
    return epoch, lr 

################################## printing, output and analysis ##################################

# Analysis by type
analysisQuestionLims = [(0,18),(19,float("inf"))]
analysisProgramLims = [(0,12),(13,float("inf"))]

toArity = lambda instance: instance["programSeq"][-1].split("_", 1)[0]
toType = lambda instance: instance["programSeq"][-1].split("_", 1)[1]

def fieldLenIsInRange(field):
    return lambda instance, group: \
        (len(instance[field]) >= group[0] and
        len(instance[field]) <= group[1])

# Groups instances based on a key
def grouperKey(toKey):
    def grouper(instances):
        res = defaultdict(list)
        for instance in instances:
            res[toKey(instnace)].append(instance)
        return res
    return grouper

# Groups instances according to their match to condition
def grouperCond(groups, isIn):
    def grouper(instances):
        res = {}
        for group in groups:
            res[group] = (instance for instance in instances if isIn(instance, group))
        return res
    return grouper 

groupers = {
    "questionLength": grouperCond(analysisQuestionLims, fieldLenIsInRange("questionSeq")),
    "programLength": grouperCond(analysisProgramLims, fieldLenIsInRange("programSeq")),
    "arity": grouperKey(toArity),
    "type": grouperKey(toType)
}

# Computes average
def avg(instances, field):
    if len(instances) == 0:
        return 0.0
    return sum(instances[field]) / len(instances)

# Prints analysis of questions loss and accuracy by their group 
def printAnalysis(res):
    if config.analysisType != "":
        print("Analysis by {type}".format(type = config.analysisType))
        groups = groupers[config.analysisType](res["preds"])
        for key in groups:
            instances = groups[key]
            avgLoss = avg(instances, "loss")
            avgAcc = avg(instances, "acc")
            num = len(instances)
            print("Group {key}: Loss: {loss}, Acc: {acc}, Num: {num}".format(key, avgLoss, avgAcc, num))

# Print results for a tier
def printTierResults(tierName, res, color):
    if res is None:
        return

    print("{tierName} Loss: {loss}, {tierName} accuracy: {acc}".format(tierName = tierName,
        loss = bcolored(res["loss"], color), 
        acc = bcolored(res["acc"], color)))
    
    printAnalysis(res)

# Prints dataset results (for several tiers)
def printDatasetResults(trainRes, evalRes, extraEvalRes):
    printTierResults("Training", trainRes, "magenta")
    printTierResults("Training EMA", evalRes["evalTrain"], "red")
    printTierResults("Validation", evalRes["val"], "cyan")
    printTierResults("Extra Training EMA", extraEvalRes["evalTrain"], "red")
    printTierResults("Extra Validation", extraEvalRes["val"], "cyan")    

# Writes predictions for several tiers
def writePreds(preprocessor, evalRes, extraEvalRes):
    preprocessor.writePreds(evalRes["evalTrain"], "evalTrain")
    preprocessor.writePreds(evalRes["val"], "val")
    preprocessor.writePreds(evalRes["test"], "test")
    preprocessor.writePreds(extraEvalRes["evalTrain"], "evalTrain", "H")
    preprocessor.writePreds(extraEvalRes["val"], "val", "H")
    preprocessor.writePreds(extraEvalRes["test"], "test", "H")

############################################# session #############################################
# Initializes TF session. Sets GPU memory configuration.
def setSession():
    sessionConfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    if config.allowGrowth:
        sessionConfig.gpu_options.allow_growth = True
    if config.maxMemory < 1.0:
        sessionConfig.gpu_options.per_process_gpu_memory_fraction = config.maxMemory
    return sessionConfig

############################################## savers #############################################
# Initializes savers (standard, optional exponential-moving-average and optional for subset of variables)
def setSavers(model):
    saver = tf.train.Saver(max_to_keep = config.weightsToKeep)

    subsetSaver = None
    if config.saveSubset:
        isRelevant = lambda var: any(s in var.name for s in config.varSubset)
        relevantVars = [var for var in tf.global_variables() if isRelevant(var)]
        subsetSaver = tf.train.Saver(relevantVars, max_to_keep = config.weightsToKeep, allow_empty = True)
    
    emaSaver = None
    if config.useEMA: 
        emaSaver = tf.train.Saver(model.emaDict, max_to_keep = config.weightsToKeep)

    return {
        "saver": saver,
        "subsetSaver": subsetSaver,
        "emaSaver": emaSaver
    }

################################### restore / initialize weights ##################################
# Restores weights of specified / last epoch if on restore mod.
# Otherwise, initializes weights.  
def loadWeights(sess, saver, init):
    if config.restoreEpoch > 0 or config.restore:
        # restore last epoch only if restoreEpoch isn't set
        if config.restoreEpoch == 0:
            # restore last logged epoch
            config.restoreEpoch, config.lr = lastLoggedEpoch()
        print(bcolored("Restoring epoch {} and lr {}".format(config.restoreEpoch, config.lr),"cyan"))
        print(bcolored("Restoring weights", "blue"))
        saver.restore(sess, config.weightsFile(config.restoreEpoch))
        epoch = config.restoreEpoch
    else:
        print(bcolored("Initializing weights", "blue"))
        sess.run(init)
        logInit()
        epoch = 0

    return epoch 

###################################### training / evaluation ######################################
# Chooses data to train on (main / extra) data. 
def chooseTrainingData(data):
    trainingData = data["main"]["train"]
    alterData = None

    if config.extra:
        if config.trainExtra:
            if config.extraVal:
                trainingData = data["extra"]["val"]
            else:
                trainingData = data["extra"]["train"]                  
        if config.alterExtra:
            alterData = data["extra"]["train"]

    return trainingData, alterData

#### evaluation
# Runs evaluation on train / val / test datasets.
def runEvaluation(sess, model, data, epoch, evalTrain = True, evalTest = False, getAtt = None):
    if getAtt is None:
        getAtt = config.getAtt
    res = {"evalTrain": None, "val": None, "test": None}
    
    if data is not None:
        if evalTrain and config.evalTrain:
            res["evalTrain"] = runEpoch(sess, model, data["evalTrain"], train = False, epoch = epoch, getAtt = getAtt)

        res["val"] = runEpoch(sess, model, data["val"], train = False, epoch = epoch, getAtt = getAtt)
        
        if evalTest or config.test:
            res["test"] = runEpoch(sess, model, data["test"], train = False, epoch = epoch, getAtt = getAtt)    
        
    return res

## training conditions (comparing current epoch result to prior ones)
def improveEnough(curr, prior, lr):
    prevRes = prior["prev"]["res"]
    currRes = curr["res"]

    if prevRes is None:
        return True

    prevTrainLoss = prevRes["train"]["loss"]
    currTrainLoss = currRes["train"]["loss"]
    lossDiff = prevTrainLoss - currTrainLoss
    
    notImprove = ((lossDiff < 0.015 and prevTrainLoss < 0.5 and lr > 0.00002) or \
                  (lossDiff < 0.008 and prevTrainLoss < 0.15 and lr > 0.00001) or \
                  (lossDiff < 0.003 and prevTrainLoss < 0.10 and lr > 0.000005))
                  #(prevTrainLoss < 0.2 and config.lr > 0.000015)
    
    return not notImprove

def better(currRes, bestRes):
    return currRes["val"]["acc"] > bestRes["val"]["acc"]

############################################## data ###############################################
#### instances and batching 
# Trims sequences based on their max length.
def trim2DVectors(vectors, vectorsLengths):
    maxLength = np.max(vectorsLengths)
    return vectors[:,:maxLength]

# Trims batch based on question length.
def trimData(data):
    data["questions"] = trim2DVectors(data["questions"], data["questionLengths"])
    return data

# Gets batch / bucket size.
def getLength(data):
    return len(data["instances"])

# Selects the data entries that match the indices. 
def selectIndices(data, indices):
    def select(field, indices): 
        if type(field) is np.ndarray:
            return field[indices]
        if type(field) is list:
            return [field[i] for i in indices]
        else:
            return field
    selected = {k : select(d, indices) for k,d in data.items()}
    return selected

# Batches data into a a list of batches of batchSize. 
# Shuffles the data by default.
def getBatches(data, batchSize = None, shuffle = True):
    batches = []

    dataLen = getLength(data)
    if batchSize is None or batchSize > dataLen:
        batchSize = dataLen
    
    indices = np.arange(dataLen)
    if shuffle:
        np.random.shuffle(indices)

    for batchStart in range(0, dataLen, batchSize):
        batchIndices = indices[batchStart : batchStart + batchSize]
        # if len(batchIndices) == batchSize?
        if len(batchIndices) >= config.gpusNum:
            batch = selectIndices(data, batchIndices)
            batches.append(batch)
            # batchesIndices.append((data, batchIndices))

    return batches

#### image batches
# Opens image files.
def openImageFiles(images):
    images["imagesFile"] = h5py.File(images["imagesFilename"], "r")
    images["imagesIds"] = None
    if config.dataset == "NLVR":
        with open(images["imageIdsFilename"], "r") as imageIdsFile:
            images["imagesIds"] = json.load(imageIdsFile)  

# Closes image files.
def closeImageFiles(images): 
    images["imagesFile"].close()

# Loads an images from file for a given data batch.
def loadImageBatch(images, batch):
    imagesFile = images["imagesFile"]
    id2idx = images["imagesIds"]

    toIndex = lambda imageId: imageId
    if id2idx is not None:
        toIndex = lambda imageId: id2idx[imageId]
    imageBatch = np.stack([imagesFile["features"][toIndex(imageId)] for imageId in batch["imageIds"]], axis = 0)
    
    return {"images": imageBatch, "imageIds": batch["imageIds"]}

# Loads images for several num batches in the batches list from start index. 
def loadImageBatches(images, batches, start, num):
    batches = batches[start: start + num]
    return [loadImageBatch(images, batch) for batch in batches]

#### data alternation
# Alternates main training batches with extra data.
def alternateData(batches, alterData, dataLen):
    alterData = alterData["data"][0] # data isn't bucketed for altered data

    # computes number of repetitions
    needed = math.ceil(len(batches) / config.alterNum) 
    print(bold("Extra batches needed: %d") % needed)
    perData = math.ceil(getLength(alterData) / config.batchSize)
    print(bold("Batches per extra data: %d") % perData)
    repetitions = math.ceil(needed / perData)
    print(bold("reps: %d") % repetitions)
    
    # make alternate batches
    alterBatches = []
    for _ in range(repetitions):
        repBatches = getBatches(alterData, batchSize = config.batchSize)
        random.shuffle(repBatches)
        alterBatches += repBatches
    print(bold("Batches num: %d") + len(alterBatches))
    
    # alternate data with extra data
    curr = len(batches) - 1
    for alterBatch in alterBatches:
        if curr < 0:
            # print(colored("too many" + str(curr) + " " + str(len(batches)),"red"))
            break
        batches.insert(curr, alterBatch)
        dataLen += getLength(alterBatch)
        curr -= config.alterNum

    return batches, dataLen

############################################ threading ############################################

imagesQueue = queue.Queue(maxsize = 20) # config.tasksNum
inQueue = queue.Queue(maxsize = 1)
outQueue = queue.Queue(maxsize = 1)

# Runs a worker thread(s) to load images while training .
class StoppableThread(threading.Thread):
    # Thread class with a stop() method. The thread itself has to check
    # regularly for the stopped() condition.

    def __init__(self, images, batches): # i
        super(StoppableThread, self).__init__()
        # self.i = i
        self.images = images
        self.batches = batches
        self._stop_event = threading.Event()

    # def __init__(self, args):
    #     super(StoppableThread, self).__init__(args = args)
    #     self._stop_event = threading.Event()

    # def __init__(self, target, args):
    #     super(StoppableThread, self).__init__(target = target, args = args)
    #     self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while not self.stopped():
            try:
                batchNum = inQueue.get(timeout = 60)
                nextItem = loadImageBatches(self.images, self.batches, batchNum, int(config.taskSize / 2))
                outQueue.put(nextItem)
                # inQueue.task_done()
            except:
                pass
        # print("worker %d done", self.i)

def loaderRun(images, batches):
    batchNum = 0

    # if config.workers == 2:           
    #     worker = StoppableThread(images, batches) # i, 
    #     worker.daemon = True
    #     worker.start() 

    #     while batchNum < len(batches):
    #         inQueue.put(batchNum + int(config.taskSize / 2))
    #         nextItem1 = loadImageBatches(images, batches, batchNum, int(config.taskSize / 2))
    #         nextItem2 = outQueue.get()

    #         nextItem = nextItem1 + nextItem2
    #         assert len(nextItem) == min(config.taskSize, len(batches) - batchNum)
    #         batchNum += config.taskSize
            
    #         imagesQueue.put(nextItem)

    #     worker.stop()
    # else:
    while batchNum < len(batches):
        nextItem = loadImageBatches(images, batches, batchNum, config.taskSize)
        assert len(nextItem) == min(config.taskSize, len(batches) - batchNum)
        batchNum += config.taskSize                    
        imagesQueue.put(nextItem)

    # print("manager loader done")

########################################## stats tracking #########################################
# Computes exponential moving average.
def emaAvg(avg, value):
    if avg is None:
        return value
    emaRate = 0.98
    return avg * emaRate + value * (1 - emaRate)

# Initializes training statistics.
def initStats():
    return {
        "totalBatches": 0,
        "totalData": 0,
        "totalLoss": 0.0,
        "totalCorrect": 0,
        "loss": 0.0,
        "acc": 0.0,
        "emaLoss": None,
        "emaAcc": None,
    }

# Updates statistics with training results of a batch
def updateStats(stats, res, batch):
    stats["totalBatches"] += 1
    stats["totalData"] += getLength(batch)

    stats["totalLoss"] += res["loss"]
    stats["totalCorrect"] += res["correctNum"]

    stats["loss"] = stats["totalLoss"] / stats["totalBatches"]
    stats["acc"] = stats["totalCorrect"] / stats["totalData"]
    
    stats["emaLoss"] = emaAvg(stats["emaLoss"], res["loss"])
    stats["emaAcc"] = emaAvg(stats["emaAcc"], res["acc"])
                                                    
    return stats 

# auto-encoder ae = {:2.4f} autoEncLoss,
# Translates training statistics into a string to print
def statsToStr(stats, res, epoch, batchNum, dataLen, startTime):
    formatStr = "\reb {epoch},{batchNum} ({dataProcessed} / {dataLen:5d}), " + \
                             "t = {time} ({loadTime:2.2f}+{trainTime:2.2f}), " + \
                             "lr {lr}, l = {loss}, a = {acc}, avL = {avgLoss}, " + \
                             "avA = {avgAcc}, g = {gradNorm:2.4f}, " + \
                             "emL = {emaLoss:2.4f}, emA = {emaAcc:2.4f}; " + \
                             "{expname} {machine}/{gpu}"

    s_epoch = bcolored("{:2d}".format(epoch),"yellow")
    s_batchNum = "{:3d}".format(batchNum)
    s_dataProcessed = bcolored("{:5d}".format(stats["totalData"]),"yellow")
    s_dataLen = dataLen
    s_time = bcolored("{:2.2f}".format(time.time() - startTime),"yellow")
    s_loadTime = res["readTime"] 
    s_trainTime = res["trainTime"]
    s_lr = bold(config.lr)
    s_loss = bcolored("{:2.4f}".format(res["loss"]), "blue")
    s_acc = bcolored("{:2.4f}".format(res["acc"]),"blue")
    s_avgLoss = bcolored("{:2.4f}".format(stats["loss"]), "blue")
    s_avgAcc = bcolored("{:2.4f}".format(stats["acc"]),"red")
    s_gradNorm = res["gradNorm"]  
    s_emaLoss = stats["emaLoss"]
    s_emaAcc = stats["emaAcc"]
    s_expname = config.expName 
    s_machine = bcolored(config.dataPath[9:11],"yellow") 
    s_gpu = bcolored(config.gpus,"yellow")

    return formatStr.format(epoch = s_epoch, batchNum = s_batchNum, dataProcessed = s_dataProcessed,
                            dataLen = s_dataLen, time = s_time, loadTime = s_loadTime,
                            trainTime = s_trainTime, lr = s_lr, loss = s_loss, acc = s_acc,
                            avgLoss = s_avgLoss, avgAcc = s_avgAcc, gradNorm = s_gradNorm,
                            emaLoss = s_emaLoss, emaAcc = s_emaAcc, expname = s_expname,
                            machine = s_machine, gpu = s_gpu)

# collectRuntimeStats, writer = None,  
'''
Runs an epoch with model and session over the data.
1. Batches the data and optionally mix it with the extra alterData.
2. Start worker threads to load images in parallel to training.
3. Runs model for each batch, and gets results (e.g. loss,  accuracy).
4. Updates and prints statistics based on batch results.
5. Once in a while (every config.saveEvery), save weights. 

Args:
    sess: TF session to run with.
    
    model: model to process data. Has runBatch method that process a given batch.
    (See model.py for further details).
    
    data: data to use for training/evaluation.
    
    epoch: epoch number.

    saver: TF saver to save weights

    calle: a method to call every number of iterations (config.calleEvery)

    alterData: extra data to mix with main data while training.

    getAtt: True to return model attentions.  
'''
def runEpoch(sess, model, data, train, epoch, saver = None, calle = None, 
    alterData = None, getAtt = False):
    # train = data["train"] better than outside argument

    # initialization
    startTime0 = time.time()

    stats = initStats()
    preds = []

    # open image files
    openImageFiles(data["images"])

    ## prepare batches
    buckets = data["data"]
    dataLen = sum(getLength(bucket) for bucket in buckets)
    
    # make batches and randomize
    batches = []
    for bucket in buckets:
        batches += getBatches(bucket, batchSize = config.batchSize)
    random.shuffle(batches)

    # alternate with extra data
    if train and alterData is not None:
        batches, dataLen = alternateData(batches, alterData, dataLen)

    # start image loaders
    if config.parallel:
        loader = threading.Thread(target = loaderRun, args = (data["images"], batches))
        loader.daemon = True
        loader.start()

    for batchNum, batch in enumerate(batches):   
        startTime = time.time()

        # prepare batch 
        batch = trimData(batch)

        # load images batch
        if config.parallel:
            if batchNum % config.taskSize == 0:
                imagesBatches = imagesQueue.get()
            imagesBatch = imagesBatches[batchNum % config.taskSize] # len(imagesBatches)     
        else:
            imagesBatch = loadImageBatch(data["images"], batch)
        for i, imageId in enumerate(batch["imageIds"]):
            assert imageId == imagesBatch["imageIds"][i]   
        
        # run batch
        res = model.runBatch(sess, batch, imagesBatch, train, getAtt) 

        # update stats
        stats = updateStats(stats, res, batch)
        preds += res["preds"]

        # if config.summerize and writer is not None:
        #     writer.add_summary(res["summary"], epoch)

        sys.stdout.write(statsToStr(stats, res, epoch, batchNum, dataLen, startTime))
        sys.stdout.flush()

        # save weights
        if saver is not None:
            if batchNum > 0 and batchNum % config.saveEvery == 0:
                print("")
                print(bold("saving weights"))
                saver.save(sess, config.weightsFile(epoch))

        # calle
        if calle is not None:            
            if batchNum > 0 and batchNum % config.calleEvery == 0:
                calle()
    
    sys.stdout.write("\r")
    sys.stdout.flush()

    closeImageFiles(data["images"])

    if config.parallel:
        loader.join() # should work

    return {"loss": stats["loss"], 
            "acc": stats["acc"],
            "preds": preds
            }

'''
Trains/evaluates the model:
1. Set GPU configurations.
2. Preprocess data: reads from datasets, and convert into numpy arrays.
3. Builds the TF computational graph for the MAC model.
4. Starts a session and initialize / restores weights.
5. If config.train is True, trains the model for number of epochs:
    a. Trains the model on training data
    b. Evaluates the model on training / validation data, optionally with 
       exponential-moving-average weights.
    c. Prints and logs statistics, and optionally saves model predictions.
    d. Optionally reduces learning rate if losses / accuracies don't improve,
       and applies early stopping.
6. If config.test is True, runs a final evaluation on the dataset and print
   final results!
'''
def main():
    with open(config.configFile(), "a+") as outFile:
        json.dump(vars(config), outFile)

    # set gpus
    if config.gpus != "":
        config.gpusNum = len(config.gpus.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    tf.logging.set_verbosity(tf.logging.ERROR)

    # process data
    print(bold("Preprocess data..."))
    start = time.time()
    preprocessor = Preprocesser()
    data, embeddings, answerDict = preprocessor.preprocessData()
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    # build model
    print(bold("Building model..."))
    start = time.time()
    model = MACnet(embeddings, answerDict)
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    # initializer
    init = tf.global_variables_initializer()

    # savers
    savers = setSavers(model)
    saver, emaSaver = savers["saver"], savers["emaSaver"]

    # sessionConfig
    sessionConfig = setSession()
    
    with tf.Session(config = sessionConfig) as sess:

        # ensure no more ops are added after model is built
        sess.graph.finalize()

        # restore / initialize weights, initialize epoch variable
        epoch = loadWeights(sess, saver, init)

        if config.train:
            start0 = time.time()

            bestEpoch = epoch 
            bestRes = None
            prevRes = None

            # epoch in [restored + 1, epochs]
            for epoch in range(config.restoreEpoch + 1, config.epochs + 1):
                print(bcolored("Training epoch {}...".format(epoch), "yellow"))
                start = time.time()
                
                # train
                # calle = lambda: model.runEpoch(), collectRuntimeStats, writer
                trainingData, alterData = chooseTrainingData(data)
                trainRes = runEpoch(sess, model, trainingData, train = True, epoch = epoch, 
                    saver = saver, alterData = alterData)
                
                # save weights
                saver.save(sess, config.weightsFile(epoch))
                if config.saveSubset:
                    subsetSaver.save(sess, config.subsetWeightsFile(epoch))                   
                
                # load EMA weights 
                if config.useEMA:
                    print(bold("Restoring EMA weights"))
                    emaSaver.restore(sess, config.weightsFile(epoch))

                # evaluation                
                evalRes = runEvaluation(sess, model, data["main"], epoch)
                extraEvalRes = runEvaluation(sess, model, data["extra"], epoch, 
                    evalTrain = not config.extraVal)

                # restore standard weights
                if config.useEMA:
                    print(bold("Restoring standard weights"))
                    saver.restore(sess, config.weightsFile(epoch))

                print("")

                epochTime = time.time() - start
                print("took {:.2f} seconds".format(epochTime))

                # print results
                printDatasetResults(trainRes, evalRes, extraEvalRes)
   
                # stores predictions and optionally attention maps
                if config.getPreds:
                    print(bcolored("Writing predictions...", "white"))
                    writePreds(preprocessor, evalRes, extraEvalRes)

                logRecord(epoch, epochTime, config.lr, trainRes, evalRes, extraEvalRes)

                # update best result
                # compute curr and prior
                currRes = {"train": trainRes, "val": evalRes["val"]}
                curr = {"res": currRes, "epoch": epoch} 

                if bestRes is None or better(currRes, bestRes):
                    bestRes = currRes
                    bestEpoch = epoch
                
                prior = {"best": {"res": bestRes, "epoch": bestEpoch}, 
                         "prev": {"res": prevRes, "epoch": epoch - 1}}

                # lr reducing
                if config.lrReduce:
                    if not improveEnough(curr, prior, config.lr):
                        config.lr *= config.lrDecayRate
                        print(colored("Reducing LR to %d" % config.lr, "red"))   

                # early stopping
                if config.earlyStopping > 0:
                    if epoch - bestEpoch > config.earlyStopping:
                        break

                # update previous result
                prevRes = currRes

            # reduce epoch back to the last one we trained on
            epoch -= 1
            print("Training took {:.2f} seconds ({:} epochs)".format(time.time() - start0, 
                epoch - config.restoreEpoch))
        
        if config.finalTest:
            print("Testing on epoch {}...".format(epoch))
            
            start = time.time()
            if epoch > 0:
                if config.useEMA:
                    emaSaver.restore(sess, config.weightsFile(epoch))
                else:
                    saver.restore(sess, config.weightsFile(epoch))

            evalRes = runEvaluation(sess, model, data["main"], epoch, evalTest = True)
            extraEvalRes = runEvaluation(sess, model, data["extra"], epoch, 
                evalTrain = not config.extraVal, evalTest = True)
                        
            print("took {:.2f} seconds".format(time.time() - start))
            printDatasetResults(None, evalRes, extraEvalRes)

            print("Writing predictions...")
            writePreds(preprocessor, evalRes, extraEvalRes)

        print(bcolored("Done!","white"))

if __name__ == '__main__':
    parseArgs()    
    loadDatasetConfig[config.dataset]()        
    main()
