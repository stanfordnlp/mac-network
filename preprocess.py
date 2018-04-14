import time
import os
import random
import json
import pickle
import numpy as np
from tqdm import tqdm
from termcolor import colored
from program_translator import ProgramTranslator #
from config import config

# Print bold tex
def bold(txt):
    return colored(str(txt),attrs = ["bold"])

# Print bold and colored text 
def bcolored(txt, color):
    return colored(str(txt), color, attrs = ["bold"])

# Write a line to file
def writeline(f, line):
    f.write(str(line) + "\n")

# Write a list to file
def writelist(f, l):
    writeline(f, ",".join(map(str, l)))

# 2d list to numpy
def vectorize2DList(items, minX = 0, minY = 0, dtype = np.int):
    maxX = max(len(items), minX)
    maxY = max([len(item) for item in items] + [minY])
    t = np.zeros((maxX, maxY), dtype = dtype)
    tLengths = np.zeros((maxX, ), dtype = np.int)
    for i, item in enumerate(items):
        t[i, 0:len(item)] = np.array(item, dtype = dtype)
        tLengths[i] = len(item)
    return t, tLengths

# 3d list to numpy
def vectorize3DList(items, minX = 0, minY = 0, minZ = 0, dtype = np.int):
    maxX = max(len(items), minX)
    maxY = max([len(item) for item in items] + [minY])
    maxZ = max([len(subitem) for item in items for subitem in item] + [minZ])
    t = np.zeros((maxX, maxY, maxZ), dtype = dtype)
    tLengths = np.zeros((maxX, maxY), dtype = np.int)
    for i, item in enumerate(items):
        for j, subitem in enumerate(item):
            t[i, j, 0:len(subitem)] = np.array(subitem, dtype = dtype)
            tLengths[i, j] = len(subitem)
    return t, tLengths

'''
Encodes text into integers. Keeps dictionary between string words (symbols) 
and their matching integers. Supports encoding and decoding.
'''
class SymbolDict(object):
    def __init__(self, empty = False):
        self.padding = "<PAD>"
        self.unknown = "<UNK>"
        self.start = "<START>"
        self.end = "<END>"

        self.invalidSymbols = [self.padding, self.unknown, self.start, self.end]

        if empty:
            self.sym2id = {} 
            self.id2sym = []            
        else:
            self.sym2id = {self.padding: 0, self.unknown: 1, self.start: 2, self.end: 3} 
            self.id2sym = [self.padding, self.unknown, self.start, self.end]
        self.allSeqs = []

    def getNumSymbols(self):
        return len(self.sym2id)

    def isPadding(self, enc):
        return enc == 0

    def isUnknown(self, enc):
        return enc == 1

    def isStart(self, enc):
        return enc == 2

    def isEnd(self, enc):
        return enc == 3

    def isValid(self, enc):
        return enc < self.getNumSymbols() and enc >= len(self.invalidSymbols)

    def resetSeqs(self):
        self.allSeqs = []

    def addSeq(self, seq):
        self.allSeqs += seq

    # Call to create the words-to-integers vocabulary after (reading word sequences with addSeq).
    def createVocab(self, minCount = 0):
        counter = {}
        for symbol in self.allSeqs:
            counter[symbol] = counter.get(symbol, 0) + 1
        for symbol in counter:
            if counter[symbol] > minCount and (symbol not in self.sym2id):
                self.sym2id[symbol] = self.getNumSymbols()
                self.id2sym.append(symbol)

    # Encodes a symbol. Returns the matching integer.
    def encodeSym(self, symbol):
        if symbol not in self.sym2id:
            symbol = self.unknown
        return self.sym2id[symbol]

    '''
    Encodes a sequence of symbols.
    Optionally add start, or end symbols. 
    Optionally reverse sequence 
    '''
    def encodeSequence(self, decoded, addStart = False, addEnd = False, reverse = False):
        if reverse:
            decoded.reverse()
        if addStart:
            decoded = [self.start] + decoded
        if addEnd:
            decoded = decoded + [self.end]
        encoded = [self.encodeSym(symbol) for symbol in decoded]
        return encoded

    # Decodes an integer into its symbol 
    def decodeId(self, enc):
        return self.id2sym[enc] if enc < self.getNumSymbols() else self.unknown

    '''
    Decodes a sequence of integers into their symbols.
    If delim is given, joins the symbols using delim,
    Optionally reverse the resulted sequence 
    '''
    def decodeSequence(self, encoded, delim = None, reverse = False, stopAtInvalid = True):
        length = 0
        for i in range(len(encoded)):
            if not self.isValid(encoded[i]) and stopAtInvalid:
                break
            length += 1
        encoded = encoded[:length]

        decoded = [self.decodeId(enc) for enc in encoded]
        if reverse:
            decoded.reverse()

        if delim is not None:
            return delim.join(decoded)
        
        return decoded

'''
Preprocesses a given dataset into numpy arrays.
By calling preprocess, the class:
1. Reads the input data files into dictionary. 
2. Saves the results jsons in files and loads them instead of parsing input if files exist/
3. Initializes word embeddings to random / GloVe.
4. Optionally filters data according to given filters.
5. Encodes and vectorize the data into numpy arrays.
6. Buckets the data according to the instances length.
'''
class Preprocesser(object):
    def __init__(self):
        self.questionDict = SymbolDict()
        self.answerDict = SymbolDict(empty = True)
        self.qaDict = SymbolDict()

        self.specificDatasetDicts = None

        self.programDict = SymbolDict()
        self.programTranslator = ProgramTranslator(self.programDict, 2)
    '''
    Tokenizes string into list of symbols.

    Args:
        text: raw string to tokenize.
        ignorePuncts: punctuation to ignore
        keptPunct: punctuation to keep (as symbol)
        endPunct: punctuation to remove if appears at the end
        delim: delimiter between symbols
        clean: True to replace text in string
        replacelistPre: dictionary of replacement to perform on the text before tokanization
        replacelistPost: dictionary of replacement to perform on the text after tokanization
    '''
    # sentence tokenizer
    allPunct = ["?", "!", "\\", "/", ")", "(", ".", ",", ";", ":"]
    def tokenize(self, text, ignoredPuncts = ["?", "!", "\\", "/", ")", "("], 
        keptPuncts = [".", ",", ";", ":"], endPunct = [">", "<", ":"], delim = " ", 
        clean = False, replacelistPre = dict(), replacelistPost = dict()):

        if clean:
            for word in replacelistPre:
                origText = text
                text = text.replace(word, replacelistPre[word])
                if (origText != text):
                    print(origText)
                    print(text)
                    print("")

            for punct in endPunct:
                if text[-1] == punct:
                    print(text)
                    text = text[:-1]
                    print(text)
                    print("")

        for punct in keptPuncts:
            text = text.replace(punct, delim + punct + delim)           
        
        for punct in ignoredPuncts:
            text = text.replace(punct, "")

        ret = text.lower().split(delim)

        if clean:
            origRet = ret
            ret = [replacelistPost.get(word, word) for word in ret]
            if origRet != ret:
                print(origRet)
                print(ret)

        ret = [t for t in ret if t != ""]
        return ret


    # Read class' generated files.
    # files interface
    def readFiles(self, instancesFilename):
        with open(instancesFilename, "r") as inFile:
            instances = json.load(inFile)
        
        with open(config.questionDictFile(), "rb") as inFile:
            self.questionDict = pickle.load(inFile)

        with open(config.answerDictFile(), "rb") as inFile:
            self.answerDict = pickle.load(inFile)

        with open(config.qaDictFile(), "rb") as inFile:
            self.qaDict = pickle.load(inFile)

        return instances 
    
    '''
    Generate class' files. Save json representation of instances and
    symbols-to-integers dictionaries.  
    '''
    def writeFiles(self, instances, instancesFilename):
        with open(instancesFilename, "w") as outFile:
            json.dump(instances, outFile)

        with open(config.questionDictFile(), "wb") as outFile:
            pickle.dump(self.questionDict, outFile)

        with open(config.answerDictFile(), "wb") as outFile:
            pickle.dump(self.answerDict, outFile)

        with open(config.qaDictFile(), "wb") as outFile:
            pickle.dump(self.qaDict, outFile)

    # Write prediction json to file and optionally a one-answer-per-line output file
    def writePreds(self, res, tier, suffix = ""):
        if res is None:
            return
        preds = res["preds"]
        sortedPreds = sorted(preds, key = lambda instance: instance["index"]) 
        with open(config.predsFile(tier + suffix), "w") as outFile:
            outFile.write(json.dumps(sortedPreds))
        with open(config.answersFile(tier + suffix), "w") as outFile:
            for instance in sortedPreds:
                writeline(outFile, instance["prediction"])
    
    # Reads NLVR data entries and create a json dictionary.
    def readNLVR(self, datasetFilename, instancesFilename, train):
        instances = []
        i = 0 

        if os.path.exists(instancesFilename):
            instances = self.readFiles(instancesFilename)
        else:
            with open(datasetFilename, "r") as datasetFile:               
                for line in datasetFile:
                    instance = json.loads(line)
                    question = instance["sentence"]
                    questionSeq = self.tokenize(question, 
                        ignoredPuncts = Preprocesser.allPunct, keptPuncts = [])

                    if train or (not config.wrdEmbUnknown):
                        self.questionDict.addSeq(question)
                        self.qaDict.addSeq(question)

                    answer = instance["label"]
                    self.answerDict.addSeq([answer])
                    self.qaDict.addSeq([answer])

                    for k in range(6):
                        instances.append({
                            "question": question,
                            "questionSeq": questionSeq,
                            "answer": answer,
                            "imageId": instance["identifier"] + "-" + str(k),
                            "index": i
                            })
                        i += 1

                random.shuffle(instances)

                self.questionDict.createVocab()
                self.answerDict.createVocab()
                self.qaDict.createVocab()

                self.writeFiles(instances, instancesFilename)

        return instances

    # Reads CLEVR data entries and create a json dictionary.
    def readCLEVR(self, datasetFilename, instancesFilename, train):
        instances = []

        if os.path.exists(instancesFilename):
            instances = self.readFiles(instancesFilename)
        else:
            with open(datasetFilename, "r") as datasetFile:
                data = json.load(datasetFile)["questions"]            
            for i in tqdm(range(len(data)), desc = "Preprocessing"):
                instance = data[i]

                question = instance["question"]
                questionSeq = self.tokenize(question)

                if train or (not config.wrdEmbUnknown):
                    self.questionDict.addSeq(questionSeq)
                    self.qaDict.addSeq(questionSeq)

                answer = instance.get("answer", "yes") # DUMMY_ANSWER
                self.answerDict.addSeq([answer])
                self.qaDict.addSeq([answer])

                dummyProgram = [{"function": "FUNC", "value_inputs": [], "inputs": []}]
                program = instance.get("program", dummyProgram)
                postfixProgram = self.programTranslator.programToPostfixProgram(program)
                programSeq = self.programTranslator.programToSeq(postfixProgram)
                programInputs = self.programTranslator.programToInputs(postfixProgram, 
                    offset = 2)

                # pass other fields to instance?
                instances.append({
                        "question": question,
                        "questionSeq": questionSeq,
                        "answer": answer,
                        "imageId": instance["image_index"],
                        "program": program,
                        "programSeq": programSeq,
                        "programInputs": programInputs,
                        "index": i
                        })

            random.shuffle(instances)

            self.questionDict.createVocab()
            self.answerDict.createVocab()
            self.qaDict.createVocab()

            self.writeFiles(instances, instancesFilename)

        return instances

    '''
    Reads data in datasetFilename, and creates json dictionary.
    If instancesFilename exists, restore dictionary from this file.
    Otherwise, save created dictionary to instancesFilename.
    '''
    def readData(self, datasetFilename, instancesFilename, train):
        # data extraction
        datasetReader = {
            "CLEVR": self.readCLEVR,
            "NLVR": self.readNLVR
        }

        return datasetReader[config.dataset](datasetFilename, instancesFilename, train)

    # Reads dataset tier (train, val, test) and returns the loaded instances 
    # and image relevant filenames
    def readTier(self, tier, train):
        imagesFilename = config.imagesFile(tier)
        datasetFilename = config.datasetFile(tier)
        instancesFilename = config.instancesFile(tier)
        
        instances = self.readData(datasetFilename, instancesFilename, train)

        images = {"imagesFilename": imagesFilename}
        if config.dataset == "NLVR":
            images["imageIdsFilename"] = config.imagesIdsFile(tier)
            
        return {"instances": instances, "images": images, "train": train}

    '''
    Reads all tiers of a dataset (train if exists, val, test).
    Creates also evalTrain tier which will optionally be used for evaluation. 
    '''
    def readDataset(self, suffix = "", hasTrain = True):
        dataset = {"train": None, "evalTrain": None, "val": None, "test": None}        
        if hasTrain:
            dataset["train"] = self.readTier("train" + suffix, train = True)
        dataset["val"] = self.readTier("val" + suffix, train = False)
        dataset["test"] = self.readTier("test" + suffix, train = False)
        
        if hasTrain:
            dataset["evalTrain"] = {}
            for k in dataset["train"]:
                dataset["evalTrain"][k] = dataset["train"][k]
            dataset["evalTrain"]["train"] = False

        return dataset

    # Transform symbols to corresponding integers and vectorize into numpy array
    def vectorizeData(self, data):
        # if "SHARED" tie symbol representations in questions and answers 
        if config.ansEmbMod == "SHARED":
            qDict = self.qaDict
        else:
            qDict = self.questionDict

        encodedQuestions = [qDict.encodeSequence(d["questionSeq"]) for d in data]
        questions, questionsL = vectorize2DList(encodedQuestions)

        answers = np.array([self.answerDict.encodeSym(d["answer"]) for d in data])
        
        # pass the whole instances? if heavy then not good
        imageIds = [d["imageId"] for d in data]
        indices = [d["index"] for d in data]
        instances = data

        return {    "questions": questions,
                    "questionLengths": questionsL,
                    "answers": answers,
                    "imageIds": imageIds,
                    "indices": indices,
                    "instances": instances
                }

    # Separates data based on a field length
    def lseparator(self, key, lims):
        maxI = len(lims)
        def separatorFn(x):
            v = x[key]
            for i, lim in enumerate(lims):
                if len(v) < lim:
                    return i
            return maxI
        return {"separate": separatorFn, "groupsNum": maxI + 1}

    # # separate data based on a field type
    # def tseparator(self, key, types):
    #     typesNum = len(types) + 1
    #     def separatorFn(x):
    #         v = str(x[key][-1])
    #         return types.get(v, len(types))
    #     return {"separate": separatorFn, "groupsNum": typesNum}

    # # separate data based on field arity
    # def bseparator(self, key):
    #     def separatorFn(x):
    #         cond = (len(x[key][-1]) == 2)
    #         return (1 if cond else 0)
    #     return {"separate": separatorFn, "groupsNum": 2}

    # Buckets data to groups using a separator
    def bucket(self, instances, separator):
        buckets = [[] for i in range(separator["groupsNum"])]
        for instance in instances:
            bucketI = separator["separate"](instance)
            buckets[bucketI].append(instance)
        return [bucket for bucket in buckets if len(bucket) > 0]

    # Re-buckets bucket list given a seperator
    def rebucket(self, buckets, separator):
        res = []
        for bucket in buckets:
            res += self.bucket(bucket, separator)
        return res

    # Buckets data based on question / program length 
    def bucketData(self, data, noBucket = False):
        if noBucket:
            buckets = [data]
        else:
            if config.noBucket:
                buckets = [data]
            elif config.noRebucket:
                questionSep = self.lseparator("questionSeq", config.questionLims)
                buckets = self.bucket(data, questionSep)
            else:
                programSep = self.lseparator("programSeq", config.programLims)
                questionSep = self.lseparator("questionSeq", config.questionLims)
                buckets = self.bucket(data, programSep)
                buckets = self.rebucket(buckets, questionSep)
        return buckets

    ''' 
    Prepares data: 
    1. Filters data according to above arguments.
    2. Takes only a subset of the data based on config.trainedNum / config.testedNum
    3. Buckets data according to question / program length
    4. Vectorizes data into numpy arrays
    '''
    def prepareData(self, data, train, filterKey = None, noBucket = False):
        filterDefault = {"maxQLength": 0, "maxPLength": 0, "onlyChain": False, "filterOp": 0}

        filterTrain = {"maxQLength": config.tMaxQ, "maxPLength": config.tMaxP,
                       "onlyChain": config.tOnlyChain, "filterOp": config.tFilterOp}

        filterVal = {"maxQLength": config.vMaxQ, "maxPLength": config.vMaxP,
                     "onlyChain": config.vOnlyChain, "filterOp": config.vFilterOp}

        filters = {"train": filterTrain, "evalTrain": filterTrain, 
                   "val": filterVal, "test": filterDefault}

        if filterKey is None:
            fltr = filterDefault
        else:
            fltr = filters[filterKey]

        # split data when finetuning on validation set 
        if config.trainExtra and config.extraVal and (config.finetuneNum > 0):
            if train:
                data = data[:config.finetuneNum]
            else: 
                data = data[config.finetuneNum:]

        typeFilter = config.typeFilters[fltr["filterOp"]]
        # filter specific settings
        if fltr["onlyChain"]:
            data = [d for d in data if all((len(inputNum) < 2) for inputNum in d["programInputs"])]
        if fltr["maxQLength"] > 0:
            data = [d for d in data if len(d["questionSeq"]) <= fltr["maxQLength"]]
        if fltr["maxPLength"] > 0:
            data = [d for d in data if len(d["programSeq"]) <= fltr["maxPLength"]]
        if len(typeFilter) > 0:
            data = [d for d in data if d["programSeq"][-1] not in typeFilter]

        # run on subset of the data. If 0 then use all data 
        num = config.trainedNum if train else config.testedNum
        # retainVal = True to retain same sample of validation across runs  
        if (not train) and (not config.retainVal):
            random.shuffle(data)
        if num > 0:
            data = data[:num]
        # set number to match dataset size 
        if train:
            config.trainedNum = len(data)
        else:
            config.testedNum = len(data)

        # bucket
        buckets = self.bucketData(data, noBucket = noBucket)
        
        # vectorize
        return [self.vectorizeData(bucket) for bucket in buckets]

    # Prepares all the tiers of a dataset. See prepareData method for further details.
    def prepareDataset(self, dataset, noBucket = False):
        if dataset is None:
            return None

        for tier in dataset:
            if dataset[tier] is not None:
                dataset[tier]["data"] = self.prepareData(dataset[tier]["instances"], 
                    train = dataset[tier]["train"], filterKey = tier, noBucket = noBucket)
        
        for tier in dataset:
            if dataset[tier] is not None:
                del dataset[tier]["instances"]

        return dataset

    # Initializes word embeddings to random uniform / random normal / GloVe. 
    def initializeWordEmbeddings(self, wordsDict = None, noPadding = False):
        # default dictionary to use for embeddings
        if wordsDict is None:
            wordsDict = self.questionDict

        # uniform initialization
        if config.wrdEmbUniform:
            lowInit = -1.0 * config.wrdEmbScale
            highInit = 1.0 * config.wrdEmbScale
            embeddings = np.random.uniform(low = lowInit, high = highInit, 
                size = (wordsDict.getNumSymbols(), config.wrdEmbDim))
        # normal initialization
        else:
            embeddings = config.wrdEmbScale * np.random.randn(wordsDict.getNumSymbols(), 
                config.wrdEmbDim)

        # if wrdEmbRandom = False, use GloVE
        counter = 0
        if (not config.wrdEmbRandom): 
            with open(config.wordVectorsFile, 'r') as inFile:
                for line in inFile:
                    line = line.strip().split()
                    word = line[0].lower()
                    vector = [float(x) for x in line[1:]]
                    index = wordsDict.sym2id.get(word)
                    if index is not None:
                        embeddings[index] = vector
                        counter += 1
        
        print(counter)
        print(self.questionDict.sym2id)
        print(len(self.questionDict.sym2id))
        print(self.answerDict.sym2id)      
        print(len(self.answerDict.sym2id))
        print(self.qaDict.sym2id)      
        print(len(self.qaDict.sym2id))

        if noPadding:            
            return embeddings # no embedding for padding symbol
        else:
            return embeddings[1:]

    '''
    Initializes words embeddings for question words and optionally for answer words
    (when config.ansEmbMod == "BOTH"). If config.ansEmbMod == "SHARED", tie embeddings for
    question and answer same symbols. 
    '''
    def initializeQAEmbeddings(self):
        # use same embeddings for questions and answers
        if config.ansEmbMod == "SHARED":
            qaEmbeddings = self.initializeWordEmbeddings(self.qaDict)
            ansMap = np.array([self.qaDict.sym2id[sym] for sym in self.answerDict.id2sym])
            embeddings = {"qa": qaEmbeddings, "ansMap": ansMap}
        # use different embeddings for questions and answers
        else:
            qEmbeddings = self.initializeWordEmbeddings(self.questionDict)
            aEmbeddings = None
            if config.ansEmbMod == "BOTH":
                aEmbeddings = self.initializeWordEmbeddings(self.answerDict, noPadding = True)
            embeddings = {"q": qEmbeddings, "a": aEmbeddings}
        return embeddings

    '''
    Preprocesses a given dataset into numpy arrays:
    1. Reads the input data files into dictionary. 
    2. Saves the results jsons in files and loads them instead of parsing input if files exist/
    3. Initializes word embeddings to random / GloVe.
    4. Optionally filters data according to given filters.
    5. Encodes and vectorize the data into numpy arrays.
    5. Buckets the data according to the instances length.
    '''
    def preprocessData(self, debug = False):
        # Read data into json and symbols' dictionaries
        print(bold("Loading data..."))
        start = time.time()
        mainDataset = self.readDataset(hasTrain = True)

        extraDataset = None
        if config.extra:
            # compositionalClevr doesn't have training dataset
            extraDataset = self.readDataset(suffix = "H", hasTrain = (not config.extraVal))          
            # extra dataset uses the same images
            if not config.extraVal:
                for tier in extraData:
                    extraDataset[tier]["images"] = mainDataset[tier]["images"]

        print("took {:.2f} seconds".format(time.time() - start))

        # Initialize word embeddings (random / glove)
        print(bold("Loading word vectors..."))
        start = time.time()
        embeddings = self.initializeQAEmbeddings()
        print("took {:.2f} seconds".format(time.time() - start))

        # Prepare data: filter, bucket, and vectorize into numpy arrays
        print(bold("Vectorizing data..."))
        start = time.time()       

        mainDataset = self.prepareDataset(mainDataset)
        # don't bucket for alternated data and also for humans data (small dataset)
        extraDataset = self.prepareDataset(extraDataset, 
            noBucket = (not config.extraVal) or (not config.alterExtra))

        data = {"main": mainDataset, "extra": extraDataset}
        print("took {:.2f} seconds".format(time.time() - start))

        config.questionWordsNum = self.questionDict.getNumSymbols()
        config.answerWordsNum = self.answerDict.getNumSymbols()
        
        return data, embeddings, self.answerDict
