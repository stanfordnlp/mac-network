import time
import os
import random
import json
import pickle
import numpy as np
from tqdm import tqdm
from termcolor import colored
from program_translator import ProgramTranslator
from config import config
from nltk.tokenize import word_tokenize
import re
import math
from collections import defaultdict

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
            self.sym2id = {self.padding: 0} 
            self.id2sym = [self.padding]            
        else:
            self.sym2id = {self.padding: 0, self.unknown: 1, self.start: 2, self.end: 3} 
            self.id2sym = [self.padding, self.unknown, self.start, self.end]
        self.allSeqs = []

    def getNumSymbols(self):
        return len(self.sym2id)

    def isValid(self, enc):
        return enc not in self.invalidSymbols

    def resetSeqs(self):
        self.allSeqs = []

    def addSymbols(self, seq):
        if type(seq) is not list:
            seq = [seq]
        self.allSeqs += seq

    # Call to create the words-to-integers vocabulary after (reading word sequences with addSymbols). 
    def addToVocab(self, symbol):
        if symbol not in self.sym2id:
            self.sym2id[symbol] = self.getNumSymbols()
            self.id2sym.append(symbol)

    # create vocab only if not existing..?
    def createVocab(self, minCount = 0, top = 0, addUnk = False, weights = False):
        counter = {}
        for symbol in self.allSeqs:
            counter[symbol] = counter.get(symbol, 0) + 1
        
        isTop = lambda symbol: True
        if top > 0:
            topItems = sorted(counter.items(), key = lambda x: x[1], reverse = True)[:top]
            tops = [k for k,v in topItems]
            isTop = lambda symbol: symbol in tops 

        if addUnk:
            self.addToVocab(self.unknown)

        for symbol in counter:
            if counter[symbol] > minCount and isTop(symbol):
                self.addToVocab(symbol)

        self.counter = counter

        self.counts = np.array([counter.get(sym,0) for sym in self.id2sym])
        
        if weights:
            self.weights = np.array([1.0 for sym in self.id2sym])
            if config.ansWeighting:
                weight = lambda w:  1.0 / float(w) if w > 0 else 0.0
                self.weights = np.array([weight(counter.get(sym,0)) for sym in self.id2sym])
                totalWeight = np.sum(self.weights)
                self.weights /= totalWeight
                self.weights *= len(self.id2sym)
            elif config.ansWeightingRoot:
                weight = lambda w:  1.0 / math.sqrt(float(w)) if w > 0 else 0
                self.weights = np.array([weight(counter.get(sym,0)) for sym in self.id2sym])
                totalWeight = np.sum(self.weights)
                self.weights /= totalWeight
                self.weights *= len(self.id2sym)
    
    # Encodes a symbol. Returns the matching integer.
    def encodeSym(self, symbol):
        if symbol not in self.sym2id:
            symbol = self.unknown
        return self.sym2id[symbol] # self.sym2id.get(symbol, None) # # -1 VQA MAKE SURE IT DOESNT CAUSE BUGS

    '''
    Encodes a sequence of symbols.
    Optionally add start, or end symbols. 
    Optionally reverse sequence 
    '''
    def encodeSeq(self, decoded, addStart = False, addEnd = False, reverse = False):
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
    def decodeSeq(self, encoded, delim = None, reverse = False, stopAtInvalid = True):
        length = 0
        for i in range(len(encoded)):
            if not self.isValid(self.decodeId(encoded[i])) and stopAtInvalid:
            #if not self.isValid(encoded[i]) and stopAtInvalid:
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
        self.loadVocabs()

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
    fullPunct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", 
                    "+", "\\", "_", "-",">", "<", "@", "`", ",", "?", "!", "%", 
                    "^", "&", "*", "~", "#", "$"]
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                     "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                     "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                     "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                     "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                     "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                     "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                     "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                     "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                     "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                     "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                     "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                     "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                     "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                     "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                     "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                     "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                     "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                     "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                     "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                     "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                     "youll": "you'll", "youre": "you're", "youve": "you've"}
    nums = { "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    articles = {"a": "", "an": "", "the": ""}
    allReplaceQ = {}
    for replace in [contractions, nums, articles]: # , 
        allReplaceQ.update(replace)
    
    allReplaceA = {}
    for replace in [contractions, nums]: # , 
        allReplaceA.update(replace)

    periodStrip = lambda self, s: re.sub(r"(?!<=\d)(\.)(?!\d)", " ", s) # :,' ?
    collonStrip = lambda self, s: re.sub(r"(?!<=\d)(:)(?!\d)", " ", s) # replace with " " or ""?
    commaNumStrip = lambda self, s: re.sub(r"(\d)(\,)(\d)", r"\1\3", s)

    # remove any non a-zA-Z0-9?
    vqaProcessText = lambda self, text, tokenize, question: self.processText(text, ignoredPunct = [], keptPunct = [], 
        endPunct = [], delimPunct = self.fullPunct, replacelistPost = self.allReplaceQ if question else self.allReplaceA, reClean = True, tokenize = tokenize)

    def processText(self, text, ignoredPunct = ["?", "!", "\\", "/", ")", "("], 
        keptPunct = [".", ",", ";", ":"], endPunct = [">", "<", ":"], delimPunct = [],
        delim = " ", clean = False, replacelistPre = dict(), replacelistPost = dict(),
        reClean = False, tokenize = True):

        if reClean:
            text = self.periodStrip(text)
            text = self.collonStrip(text)
            text = self.commaNumStrip(text)

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

        for punct in keptPunct:
            text = text.replace(punct, delim + punct + delim)           
        
        for punct in ignoredPunct:
            text = text.replace(punct, "")

        for punct in delimPunct:
            text = text.replace(punct, delim)

        text = text.lower()

        if config.tokenizer == "stanford":
            ret = StanfordTokenizer().tokenize(text)
        elif config.tokenizer == "nltk":
            ret = word_tokenize(text)
        else:    
            ret = text.split() # delim

        ret = [replacelistPost.get(word, word) for word in ret]

        ret = [t for t in ret if t != ""]
        if not tokenize:
            ret = delim.join(ret)

        return ret

    # Read class generated files.
    # files interface
    def readInstances(self, instancesFilename):
        with open(instancesFilename, "r") as inFile:
            instances = json.load(inFile)
        return instances

    '''
    Generate class' files. Save json representation of instances and
    symbols-to-integers dictionaries.  
    '''
    def writeInstances(self, instances, instancesFilename):
        with open(instancesFilename, "w") as outFile:
            json.dump(instances, outFile)

    def setVocabs(self):
        self.createVocabs()
        self.writeVocabs()

    def createVocabs(self):
        ansAddUnk = True
        self.questionDict.createVocab(minCount = config.wrdEmbQMinCount, top = config.wrdEmbQTop)
        self.answerDict.createVocab(minCount = config.wrdEmbAMinCount, top = config.wrdEmbATop, addUnk = ansAddUnk, weights = True) # config
        self.qaDict.createVocab(minCount = config.wrdEmbQMinCount)

    def loadVocabs(self):
        if os.path.exists(config.qaDictFile()):
            print("load dictionaries")
            with open(config.questionDictFile(), "rb") as inFile:
                self.questionDict = pickle.load(inFile)

            with open(config.answerDictFile(), "rb") as inFile:
                self.answerDict = pickle.load(inFile)

            with open(config.qaDictFile(), "rb") as inFile:
                self.qaDict = pickle.load(inFile)

    def writeVocabs(self):
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
        
        answersFile = config.answersFile(tier + suffix)
        if config.dataset == "CLEVR":    
            with open(answersFile, "w") as outFile:
                for instance in sortedPreds:
                    writeline(outFile, instance["prediction"])
        else:
            with open(answersFile, "w") as outFile:
                results = [{"question_id": instance.get("questionId", "NONE"), 
                            "answer": instance["prediction"]} 
                        for instance in sortedPreds]
                outFile.write(json.dumps(results))
    
    # Reads NLVR data entries and create a json dictionary.
    def readNLVR(self, datasetFilename, instancesFilename, tier, train, imageIndex):
        instances = []
        i = 0 

        if os.path.exists(instancesFilename):
            instances = self.readInstacnes(instancesFilename)
        else:
            with open(datasetFilename, "r") as datasetFile:               
                for line in datasetFile:
                    instance = json.loads(line)
                    questionStr = instance["sentence"]
                    question = self.processText(questionStr, 
                        ignoredPunct = Preprocesser.allPunct, keptPunct = [])

                    if train or (not config.wrdEmbQUnk):
                        self.questionDict.addSymbols(question)
                        self.qaDict.addSymbols(question)

                    answer = instance["label"]
                    if train or (not config.wrdEmbAUnk):
                        self.answerDict.addSymbols(answer)
                        self.qaDict.addSymbols(answer)

                    imageId = instance["identifier"] + "-" + str(k)
                    imageIdx = imageIndex[imageId]["idx"]
                    # int(imageId) if imageIndex is None else imageIndex[imageId]["idx"]
                    
                    for k in range(6):
                        instances.append({
                            "questionStr": questionStr,
                            "question": question,
                            "answer": answer,
                            "imageId": {"group": tier, "id": imageId, "idx": imageIdx}, # imageInfo[imageId]["idx"]
                            "tier": tier,
                            "index": i
                            })
                        i += 1

                random.shuffle(instances)

                self.writeInstances(instances, instancesFilename)

        return instances

    def readVQA(self, datasetFilename, instancesFilename, tier, updateVocab, imageIndex = None):
        vocabq = set(json.load(open("newqVocabFileVQA.json")))
        vocaba = set(json.load(open("newaVocabFileVQA.json")))
        counterq = 0
        countera = 0        
        instances = []
        qId2idx = {}
        annotationsFilename = config.annotationsFile(tier)
        pairsFilename = config.pairsFile(tier)
        
        if os.path.exists(instancesFilename):
            instances = self.readInstances(instancesFilename)
        else:
            with open(datasetFilename[0], "r") as questionsFile:
                questions = json.load(questionsFile)["questions"]            
            
            index = 0

            for i in tqdm(range(len(questions)), desc = "Preprocessing"):
                instance = questions[i]

                questionStr = instance["question"]
                question = self.vqaProcessText(questionStr, True, True)

                qlist = question + ["?", " ", " ", " "]

                if config.questionLim > 0 and len(question) > config.questionLim:
                    continue

                if updateVocab or (not config.wrdEmbQUnk):
                    self.questionDict.addSymbols(question)
                    self.qaDict.addSymbols(question)

                choices, choiceStrs = None, None
                if config.ansFormat == "mc":
                    choiceStrs = instance["multiple_choices"]
                    choices = []
                    for choiceStr in choiceStrs:
                        choice = self.vqaProcessText(choiceStr, config.ansTokenize, False)
                        if updateVocab or (not config.wrdEmbAUnk):
                            self.answerDict.addSymbols(choice)
                            self.qaDict.addSymbols(choice)
                        choices.append(choice)
                    choices = list(set(choices))    
                    
                    if len(choices) != len(choiceStrs):
                        print(choiceStrs)
                        print(choices)

                imageId = instance["image_id"]
                imageInfo = imageIndex[str(imageId)]

                ccounts = {"verbs": 0, "nouns": 0, "adj": 0, "preps": 0, "conj": 0}
                trans = {"VERB": "verbs", "NOUN": "nouns", "ADJ": "adj", "ADP": "preps", "CONJ": "conj"}

                if all([(x in vocabq) for x in questionStr[:-1].split()]):
                    counterq += 1

                instances.append({
                    "questionStr": questionStr,
                    "question": question,
                    "questionId": instance["question_id"],
                    "answer": "yes" if config.ansFormat == "oe" else 0, # Dummy answer
                    "answerFreq": ["yes"], # Dummy answer
                    "imageId": {"group": tier, "id": imageId, "idx": imageInfo["idx"]},
                    "choiceStrs": choiceStrs,
                    "choices": choices,              
                    "tier": tier,
                    "index": index
                })               

                if config.imageObjects:
                    instances[-1]["objectsNum"] = imageInfo["objectsNum"]

                qId2idx[instance["question_id"]] = index
                index += 1

            if tier != "test":
                with open(annotationsFilename, "r") as annotationsFile:
                    annotations = json.load(annotationsFile)["annotations"]            
                for i in tqdm(range(len(annotations)), desc = "Preprocessing"):
                    instance = annotations[i]
                    if instance["question_id"] not in qId2idx:
                        continue
                    idx = qId2idx[instance["question_id"]]
                    answerStr = instance["multiple_choice_answer"]
                    answer = self.vqaProcessText(answerStr, config.ansTokenize, False)
                    if config.ansFormat == "mc":
                        answer = instances[idx]["choices"].index(answer)
                    
                    answerFreqStrs = []
                    answerFreq = []

                    if instance["multiple_choice_answer"] in vocaba:
                        countera += 1

                    for answerData in instance["answers"]:
                        answerStr = answerData["answer"]
                        answer = self.vqaProcessText(answerStr, config.ansTokenize, False)
                        if updateVocab or (not config.wrdEmbAUnk):
                            self.answerDict.addSymbols(answer)
                            self.qaDict.addSymbols(answer)
                        answerFreqStrs.append(answerStr)
                        answerFreq.append(answer)

                    instances[idx].update({
                            "answerStr": answerStr,
                            "answer": answer,
                            "answerFreqStrs": answerFreqStrs,
                            "answerFreq": answerFreq,
                            "questionType": instance["question_type"],
                            "answerType": instance["answer_type"]
                            })

                if config.dataVer == 2:
                    with open(pairsFilename, "r") as pairsFile:
                        pairs = json.load(pairsFile)
                    for pair in pairs:
                        if pair[0] in qId2idx: 
                            instances[qId2idx[pair[0]]]["complementary"] = qId2idx.get(pair[1],None)
                        if pair[1] in qId2idx:
                            instances[qId2idx[pair[1]]]["complementary"] = qId2idx.get(pair[0],None)

            random.shuffle(instances)
            self.writeInstances(instances, instancesFilename)

        return instances       

    def filterUnk(self, dataset, tier, filterInstances):
        print("filtering unknown answers " + tier)
        totalScore = 0.0
        numQuestions = float(len(dataset["instances"]))
        for instance in dataset["instances"]:
            instance["answerFreq"] = [answer for answer in instance["answerFreq"] if 
                answer in self.answerDict.sym2id]

            answersSet = {}
            for answer in instance["answerFreq"]:
                answersSet[answer] = answersSet.get(answer, 0) + 1

            bestAnswer = "<>"
            bestCount = 0
            for answer in answersSet:
                if answersSet[answer] > bestCount:
                    bestAnswer = answer
                    bestCount = answersSet[answer]

            totalScore += min(bestCount * 0.3, 1)

        print("max score {}".format(totalScore / numQuestions))

        if filterInstances:
            if config.lossType in ["softmax", "svm", "probSoftmax"]:
                dataset["instances"] = [instance for instance in dataset["instances"] if 
                    instance["answer"] in self.answerDict.sym2id]
            # else:
                dataset["instances"] = [instance for instance in dataset["instances"] if 
                    len(instance["answerFreq"]) > 0]

    # Reads CLEVR data entries and create a json dictionary.
    def readVG(self, datasetFilename, instancesFilename, tier, updateVocab, imageIndex = None):
        instances = []

        if os.path.exists(instancesFilename):
            instances = self.readInstances(instancesFilename)
        else:
            with open(datasetFilename[0], "r") as datasetFile:
                data = json.load(datasetFile)
            for i in tqdm(range(len(data)), desc = "Preprocessing"):
                instance = data[i]
                for q in instance["qas"]:
                    questionStr = q["question"]
                    question = self.processText(questionStr)

                    if updateVocab or (not config.wrdEmbQUnk):
                        self.questionDict.addSymbols(question)
                        self.qaDict.addSymbols(question)

                    answer = instance.get("answer", "yes") # DUMMY_ANSWER
                    
                    if updateVocab or (not config.wrdEmbAUnk):
                        self.answerDict.addSymbols(answer)
                        self.qaDict.addSymbols(answer)

                    # pass other fields to instance?
                    instances.append({
                            "questionStr": questionStr,
                            "question": question,
                            "answer": answer,
                            "imageId": {"group": tier, "id": 0, "idx": 0},
                            "tier": tier,
                            "index": i
                            })

            random.shuffle(instances)

            self.writeInstances(instances, instancesFilename)

        return instances

    def readV7W(self, datasetFilename, instancesFilename, tier, updateVocab, imageIndex = None):
        instances = []

        if os.path.exists(instancesFilename):
            instances = self.readInstances(instancesFilename)
        else:
            with open(datasetFilename[0], "r") as datasetFile:
                data = json.load(datasetFile)["images"]
            for i in tqdm(range(len(data)), desc = "Preprocessing"):
                instance = data[i]
                for q in instance["qa_pairs"]:
                    questionStr = q["question"]
                    question = self.processText(questionStr)

                    if updateVocab or (not config.wrdEmbQUnk):
                        self.questionDict.addSymbols(question)
                        self.qaDict.addSymbols(question)

                    answer = instance.get("answer", "yes") # DUMMY_ANSWER
                    
                    if updateVocab or (not config.wrdEmbAUnk):
                        self.answerDict.addSymbols(answer)
                        self.qaDict.addSymbols(answer)

                    instances.append({
                            "questionStr": questionStr,
                            "question": question,
                            "answer": answer,
                            "imageId": {"group": tier, "id": instance["image_id"], "idx": instance["image_id"]},
                            "tier": tier,
                            "index": i
                            })

            random.shuffle(instances)

            self.writeInstances(instances, instancesFilename)

        return instances

    def readCLEVR(self, datasetFilename, instancesFilename, tier, updateVocab, imageIndex = None):
        instances = []

        if os.path.exists(instancesFilename):
            instances = self.readInstances(instancesFilename)
        else:
            with open(datasetFilename[0], "r") as datasetFile:
                data = json.load(datasetFile)["questions"]            
            for i in tqdm(range(len(data)), desc = "Preprocessing"):
                instance = data[i]

                questionStr = instance["question"]
                question = self.processText(questionStr)

                if updateVocab or (not config.wrdEmbQUnk):
                    self.questionDict.addSymbols(question)
                    self.qaDict.addSymbols(question)

                answer = instance.get("answer", "yes") # DUMMY_ANSWER
                
                if updateVocab or (not config.wrdEmbAUnk):
                    self.answerDict.addSymbols(answer)
                    self.qaDict.addSymbols(answer)

                dummyProgram = [{"function": "FUNC", "value_inputs": [], "inputs": []}]
                program = instance.get("program", dummyProgram)
                postfixProgram = self.programTranslator.programToPostfixProgram(program)
                programSeq = self.programTranslator.programToSeq(postfixProgram)
                programInputs = self.programTranslator.programToInputs(postfixProgram, 
                    offset = 2)

                instances.append({
                        "questionStr": questionStr,
                        "question": question,
                        "answer": answer,
                        "imageId": {"group": tier, "id": instance["image_index"], "idx": instance["image_index"]},
                        "program": program,
                        "programSeq": programSeq,
                        "programInputs": programInputs,
                        "tier": tier,
                        "index": i
                        })

            random.shuffle(instances)

            self.writeInstances(instances, instancesFilename)

        return instances

    def readGQA(self, datasetFilename, instancesFilename, tier, updateVocab, imageIndex = None):
        instances = []
        if os.path.exists(instancesFilename):
            instances = self.readInstances(instancesFilename)
        else:
            data = []
            datal1, datal2 = None, None
            for vf in datasetFilename:
                with open(vf, "r") as datasetFile:
                    data += json.load(datasetFile)["questions"]
                    if (datal1 is None):
                        datal1 = len(data)
                    if datal2 is None:
                        datal2 = len(data)

            for i in tqdm(range(len(data)), desc = "Preprocessing"):
                instance = data[i]

                questionStr = instance["question"]
                question = self.processText(questionStr)

                qlist = question + ["?", " ", " ", " "]

                if (i < datal1) and (updateVocab or (not config.wrdEmbQUnk)):
                    self.questionDict.addSymbols(question)
                    self.qaDict.addSymbols(question)

                answer = instance.get("answer", "yes") # DUMMY_ANSWER
                
                if (i < datal2) and (updateVocab or (not config.wrdEmbAUnk)):
                    self.answerDict.addSymbols(answer)
                    self.qaDict.addSymbols(answer)

                imageId = instance["imageId"]
                imageInfo = imageIndex[str(imageId)]

                instances.append({
                        "questionStr": questionStr,
                        "question": question,
                        "answer": answer,
                        "imageId": {"group": tier, "id": imageId, "idx": imageInfo["index"]},
                        "tier": tier,
                        "index": i,
                        "questionId": instance["questionId"]
                        })

                if config.imageObjects:
                    instances[-1]["objectsNum"] = imageInfo["objectsNum"]

                if "type" in instance:
                    instances[-1]["type"] = instance["type"]
                if "group" in instance:
                    instances[-1]["group"] = instance["group"]

            random.shuffle(instances)

            self.writeInstances(instances, instancesFilename)

        return instances

    def encodeQuestionStr(self, questionStr):
        qDict = self.qaDict if config.ansEmbMod == "SHARED" else self.questionDict
        question = self.vqaProcessText(questionStr, True, True)
        return question

    '''
    Reads data in datasetFilename, and creates json dictionary.
    If instancesFilename exists, restore dictionary from this file.
    Otherwise, save created dictionary to instancesFilename.
    '''
    def readData(self, datasetFilename, instancesFilename, tier, updateVocab, imageIndex):
        # data extraction
        datasetReader = {
            "VG": self.readVG,
            "V7W": self.readV7W,
            "CLEVR": self.readCLEVR,
            "NLVR": self.readNLVR,
            "VQA": self.readVQA,
            "GQA": self.readGQA
        }

        return datasetReader[config.dataset](datasetFilename, instancesFilename, tier, updateVocab, imageIndex)

    # Reads dataset tier (train, val, test) and returns the loaded instances 
    # and image relevant filenames
    def readTier(self, tier, train):
        print("Reading tier {}".format(tier))
        imagesFilename = config.imagesFile(tier)
        if tier == "val" and config.valFilenames != []:
            datasetFilename = [config.datasetFile(tier)] + [config.dataFile(vf) for vf in config.valFilenames]
            instancesFilename = config.instancesFile("finalspecialVallls")
        else:
            datasetFilename = [config.datasetFile(tier)]
            instancesFilename = config.instancesFile(tier)

        imgsInfoFilename = config.imgsInfoFile(tier)        
        with open(imgsInfoFilename, "r") as file:
            imageIndex = json.load(file)  
        instances = self.readData(datasetFilename, instancesFilename, tier, train, imageIndex) # updateVocab = 

        images = {tier: {"imagesFilename": imagesFilename, "imgsInfoFilename": imgsInfoFilename}}       

        return {"instances": instances, "images": images, "train": train} # 

    '''
    Reads all tiers of a dataset (train if exists, val, test).
    Creates also evalTrain tier which will optionally be used for evaluation. 
    '''
    def readDataset(self, suffix = "", hasTrain = True, trainOnVal = False):
        dataset = {"train": None, "evalTrain": None, "val": None, "test": None}
        if hasTrain:
            dataset["train"] = self.readTier("train" + suffix, train = True)

        dataset["val"] = self.readTier("testdev" + suffix, train = trainOnVal)
        dataset["test"] = self.readTier("val" + suffix, train = False) # swapped testdev and val to get testdev results during training.

        if hasTrain:
            dataset["evalTrain"] = {}
            for k in dataset["train"]:
                dataset["evalTrain"][k] = dataset["train"][k]
            dataset["evalTrain"]["train"] = False

        if trainOnVal:
            trainLen = len(dataset["train"]["instances"])
            for instance in dataset["val"]["instances"]:
                instance["index"] += trainLen
                if "complementary" in instance:
                    instance["complementary"] += trainLen
            if config.valSplit > 0:
                dataset["train"]["instances"] += dataset["val"]["instances"][config.valSplit:]
                dataset["val"]["instances"] = dataset["val"]["instances"][:config.valSplit]
                print(len(dataset["train"]["instances"]))
                print(len(dataset["val"]["instances"]))
            dataset["train"]["images"].update(dataset["val"]["images"])

        self.setVocabs()

        if config.dataset == "VQA" and config.ansFormat == "oe": # and (not config.ansAddUnk):
            self.filterUnk(dataset["train"], "train", filterInstances = True)
            self.filterUnk(dataset["val"], "val", filterInstances = False)
            self.filterUnk(dataset["evalTrain"], "evalTrain", filterInstances = False)

        return dataset

    # Transform symbols to corresponding integers and vectorize into numpy array
    def vectorizeData(self, data):
        if config.ansEmbMod == "SHARED":
            qDict, aDict = self.qaDict, self.qaDict
        else:
            qDict, aDict = self.questionDict, self.answerDict
        oeAnswers = self.answerDict

        encodedQuestions = [qDict.encodeSeq(d["question"]) for d in data]
        questions, questionsL = vectorize2DList(encodedQuestions)

        if config.ansFormat == "mc":
            answers = np.array([d["answer"] for d in data])
        else:
            answers = np.array([oeAnswers.encodeSym(d["answer"]) for d in data])
        
        imageIds = [d["imageId"] for d in data]

        indices = [d["index"] for d in data]
        instances = data

        vectorizedData = {    
            "questions": questions,
            "questionLengths": questionsL,
            "answers": answers,              
            "imageIds": imageIds,
            "indices": indices,
            "instances": instances
        }

        if config.imageObjects:
            vectorizedData["objectsNums"] = [d["objectsNum"] for d in data]

        if config.dataset == "VQA":
            encodedAnswerFreqs = [oeAnswers.encodeSeq(d["answerFreq"]) for d in data]
            answerFreqs, answerFreqsN = vectorize2DList(encodedAnswerFreqs, minY = config.AnswerFreqMaxNum)

            vectorizedData.update({                  
                "answerFreqs": answerFreqs,
                "answerFreqNums": answerFreqsN   
            })

            if config.ansFormat == "mc":
                encodedChoices = [aDict.encodeSeq(d["choices"]) for d in data]
                choices, choicesN = vectorize2DList(encodedChoices, minY = config.choicesMaxNum)

                vectorizedData.update({                  
                    "choices": choices,
                    "choicesNums": choicesN  
                })      

        return vectorizedData

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
            else:
                questionSep = self.lseparator("question", config.questionLims)
                buckets = self.bucket(data, questionSep)
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
            data = [d for d in data if len(d["question"]) <= fltr["maxQLength"]]
        if fltr["maxPLength"] > 0:
            data = [d for d in data if len(d["programSeq"]) <= fltr["maxPLength"]]
        if len(typeFilter) > 0:
            data = [d for d in data if d["programSeq"][-1] not in typeFilter]

        if not train and config.testAll:
            pass
        else:
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
    def initEmbRandom(self, num, dim):
        # uniform initialization
        if config.wrdEmbUniform:
            lowInit = -1.0 * config.wrdEmbScale
            highInit = 1.0 * config.wrdEmbScale
            embeddings = np.random.uniform(low = lowInit, high = highInit, 
                size = (num, dim))
        # normal initialization
        else:
            embeddings = config.wrdEmbScale * np.random.randn(num, dim)
        return embeddings

    def sentenceEmb(self, sentence, wordVectors, dim):
        words = sentence.split(" ")
        wordEmbs = self.initEmbRandom(len(words), dim)
        for idx, word in enumerate(words):
            if word in wordVectors:
                wordEmbs[idx] = wordVectors[word]
        sentenceEmb = np.mean(wordEmbs, axis = 0)        
        return sentenceEmb

    def initializeWordEmbeddings(self, dim, wordsDict = None, random = False, name = ""):
        # default dictionary to use for embeddings
        if wordsDict is None:
            wordsDict = self.questionDict

        embsFile = config.dictNpyFile(name)
        if config.npy and os.path.exists(embsFile):
            embeddings = np.load(embsFile)
            print("loaded embs from file")
        else: 
            embeddings = self.initEmbRandom(wordsDict.getNumSymbols(), dim)

            # if wrdEmbRandom = False, use GloVE
            counter = 0
            if not random:
                wordVectors = {}
                with open(config.wordVectorsFile, "r") as inFile:
                    for line in inFile:
                        line = line.strip().split()
                        word = line[0].lower()
                        vector = np.array([float(x) for x in line[1:]])
                        wordVectors[word] = vector

                    for sym,idx in wordsDict.sym2id.items():
                        if " " in sym:
                            symEmb = self.sentenceEmb(sym, wordVectors, config.wrdAEmbDim)
                            embeddings[idx] = symEmb
                        else:
                            if sym in wordVectors:
                                embeddings[idx] = wordVectors[sym]
                                counter += 1
            
            print(counter)
            # print(self.questionDict.sym2id)
            print("q", len(self.questionDict.sym2id))
            # print(self.answerDict.sym2id)      
            print("a", len(self.answerDict.sym2id))
            # print(self.qaDict.sym2id)      
            print(len(self.qaDict.sym2id))
        
            with open("newqVocabFile.json", "w") as outFile:
                json.dump(list(self.questionDict.sym2id.keys()), outFile)

            with open("newaVocabFile.json", "w") as outFile:
                json.dump(list(self.answerDict.sym2id.keys()), outFile)

            if config.npy:
                np.save(embsFile, embeddings)                               
        
        if wordsDict.padding in wordsDict.sym2id and wordsDict != self.answerDict: #?? padding for answers?
            return embeddings[1:] # no embedding for padding symbol
        return embeddings    

    def initializeWordEmbeddingsList(self, dim, wordlist, random = False, name = ""):
        embsFile = config.dictNpyFile(name)
        if config.npy and os.path.exists(embsFile):
            embeddings = np.load(embsFile)
            print("loaded embs from file")
        else: 
            embeddings = self.initEmbRandom(len(wordlist), dim)

            # if wrdEmbRandom = False, use GloVE
            counter = 0
            if not random:
                wordVectors = {}
                with open(config.wordVectorsSemanticFile, "r") as inFile:
                    for line in inFile:
                        line = line.strip().split()
                        word = line[0].lower()
                        vector = np.array([float(x) for x in line[1:]])
                        wordVectors[word] = vector

                    # answordavg
                    for idx, sym in enumerate(wordlist):
                        if " " in sym:
                            symEmb = self.sentenceEmb(sym, wordVectors, dim)
                            embeddings[idx] = symEmb
                        else:
                            if sym in wordVectors:
                                embeddings[idx] = wordVectors[sym]
                                counter += 1
            
            print(counter)
        
            if config.npy:
                np.save(embsFile, embeddings)                               

        return embeddings 

    def initVocabs(self):
        classes = ['__background__']  
        with open(config.vocabFile("classes"), "r") as f:
            for object in f.readlines():
                classes.append(object.split(",")[0].lower().strip())
        config.classesNum = len(classes)

        attrs = ['__noAttr__']  
        with open(config.vocabFile("attrs"), "r") as f:
            for object in f.readlines():
                attrs.append(object.split(",")[0].lower().strip())
        config.attrsNum = len(attrs)

        return classes, attrs
    '''
    Initializes words embeddings for question words and optionally for answer words
    (when config.ansEmbMod == "BOTH"). If config.ansEmbMod == "SHARED", tie embeddings for
    question and answer same symbols. 
    '''
    def initializeQAEmbeddings(self):
        # use same embeddings for questions and answers
        if config.ansEmbMod == "SHARED":
            qaEmbeddings = self.initializeWordEmbeddings(config.wrdQEmbDim, self.qaDict, 
                random = config.wrdEmbQRandom, name = "qa")
            oeAnswers = np.array([self.qaDict.sym2id[sym] for sym in self.answerDict.id2sym]) # [config.answerDelta:]
            embeddings = {"qa": qaEmbeddings, "oeAnswers": oeAnswers}
        # use different embeddings for questions and answers
        else:
            qEmbeddings = self.initializeWordEmbeddings(config.wrdQEmbDim, self.questionDict, 
                random = config.wrdEmbQRandom, name = "q")
            aEmbeddings = None
            if config.ansEmbMod == "BOTH":
                aEmbeddings = self.initializeWordEmbeddings(config.wrdAEmbDim, self.answerDict, 
                    random = config.wrdEmbARandom, name = "a")
            embeddings = {"q": qEmbeddings, "a": aEmbeddings} # mask

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
        config.AnswerFreqMaxNum = 10
        config.choicesMaxNum = 18 # read it from json "num_choices"

        # Read data into json and symbols' dictionaries
        print(bold("Loading data..."))
        start = time.time()
        mainDataset = self.readDataset(hasTrain = True, trainOnVal = config.trainOnVal)

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
        # TODO: change not extraVal to a flag?
        extraDataset = self.prepareDataset(extraDataset, 
            noBucket = (not config.extraVal) or (not config.alterExtra))

        data = {"main": mainDataset, "extra": extraDataset}
        print("took {:.2f} seconds".format(time.time() - start))

        # config.questionWordsNum = self.questionDict.getNumSymbols()
        # config.answerDelta = 4
        config.answerWordsNum = self.answerDict.getNumSymbols() # - config.answerDelta
        print("answerWordsNum")
        print(config.answerWordsNum)
        return data, embeddings, self.answerDict, self.questionDict
