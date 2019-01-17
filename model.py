import time
import math
import numpy as np
import tensorflow as tf

import ops
from config import config
from mac_cell import MACCell
'''
The MAC network model. It performs reasoning processes to answer a question over
knowledge base (the image) by decomposing it into attention-based computational steps,
each perform by a recurrent MAC cell.

The network has three main components. 
Input unit: processes the network inputs: raw question strings and image into
distributional representations.

The MAC network: calls the MACcells (mac_cell.py) config.netLength number of times,
to perform the reasoning process over the question and image.

The output unit: a classifier that receives the question and final state of the MAC
network and uses them to compute log-likelihood over the possible one-word answers.       
'''
class MACnet(object):

    '''Initialize the class.
    
    Args: 
        embeddingsInit: initialization for word embeddings (random / glove).
        answerDict: answers dictionary (mapping between integer id and symbol).
    '''
    def __init__(self, embeddingsInit, answerDict, questionDict, nextElement = None):
        self.input = nextElement        
        self.embeddingsInit = embeddingsInit
        self.answerDict = answerDict
        self.questionDict = questionDict
        self.build()

    '''
    Initializes placeholders.
        questionIndicesAll: integer ids of question words. 
        [batchSize, questionLength]
        
        questionLengthsAll: length of each question. 
        [batchSize]
        
        imagesPlaceholder: image features. 
        [batchSize, channels, height, width]
        (converted internally to [batchSize, height, width, channels])
        
        answerIndicesAll: integer ids of answer words. 
        [batchSize]

        lr: learning rate (tensor scalar)
        train: train / evaluation (tensor boolean)

        dropout values dictionary (tensor scalars)
    '''
    # change to H x W x C?

    def addPlaceholders(self):
        with tf.variable_scope("Placeholders"):
            ## data
            # questions            
            self.questionIndicesAll = tf.placeholder(tf.int32, shape = (None, None))
            self.questionLengthsAll = tf.placeholder(tf.int32, shape = (None, ))

            # images
            # put image known dimension as last dim?
            if config.imageObjects:
                self.imagesAll = tf.placeholder(tf.float32, shape = (None, None, None))
                self.imagesObjectNumAll = tf.placeholder(tf.int32, shape = (None, ))
            else:
                self.imagesPlaceholder = tf.placeholder(tf.float32, shape = (None, None, None, None))
                self.imagesAll = tf.transpose(self.imagesPlaceholder, (0, 2, 3, 1))

            # answers
            self.answerIndicesAll = tf.placeholder(tf.int32, shape = (None, ))
            if config.dataset == "VQA":
                self.answerFreqListsAll = tf.placeholder(tf.int32, shape = (None, None))
                self.answerFreqNumsAll = tf.placeholder(tf.int32, shape = (None, ))

            if config.ansFormat == "mc":
                self.choicesIndicesAll = tf.placeholder(tf.int32, shape = (None, None))
                self.choicesNumsAll = tf.placeholder(tf.int32, shape = (None, ))
                # in general could consolidate that with mc and make it more general if i did choicesIndices all of them
                # in case of open ended

            ## optimization
            self.lr = tf.placeholder(tf.float32, shape = ())
            self.train = tf.placeholder(tf.bool, shape = ())
            self.batchSizeAll = tf.shape(self.questionIndicesAll)[0]

            ## dropouts
            # TODO: change dropouts to be 1 - current
            self.dropouts = {
                "encInput": tf.placeholder(tf.float32, shape = ()),
                "encState": tf.placeholder(tf.float32, shape = ()),
                "stem": tf.placeholder(tf.float32, shape = ()),
                "question": tf.placeholder(tf.float32, shape = ()),
                "read": tf.placeholder(tf.float32, shape = ()),
                "write": tf.placeholder(tf.float32, shape = ()),
                "memory": tf.placeholder(tf.float32, shape = ()),
                "output": tf.placeholder(tf.float32, shape = ()),
                "controlPre": tf.placeholder(tf.float32, shape = ()),
                "controlPost": tf.placeholder(tf.float32, shape = ()),
                "wordEmb": tf.placeholder(tf.float32, shape = ()),
                "word": tf.placeholder(tf.float32, shape = ()),
                "vocab": tf.placeholder(tf.float32, shape = ()),
                "object": tf.placeholder(tf.float32, shape = ()),
                "wordStandard": tf.placeholder(tf.float32, shape = ())
            }

            # batch norm params
            self.batchNorm = {"decay": config.bnDecay, "train": self.train}

            self.imageInDim = config.imageDims[-1]
            if not config.imageObjects:
                self.H, self.W, self.imageInDim = 7, 7, 2048# config.imageDims
                if config.dataset == "CLEVR":
                    self.H, self.W, self.imageInDim = 14, 14, 1024

    # Feeds data into placeholders. See addPlaceholders method for further details.
    def createFeedDict(self, data, images, train):
        feedDict = {
            self.questionIndicesAll: data["questions"],
            self.questionLengthsAll: data["questionLengths"],
            self.answerIndicesAll: data["answers"],        
            self.dropouts["encInput"]: config.encInputDropout if train else 1.0,
            self.dropouts["encState"]: config.encStateDropout if train else 1.0,
            self.dropouts["stem"]: config.stemDropout if train else 1.0,
            self.dropouts["question"]: config.qDropout if train else 1.0, #_
            self.dropouts["memory"]: config.memoryDropout if train else 1.0,
            self.dropouts["read"]: config.readDropout if train else 1.0, #_
            self.dropouts["write"]: config.writeDropout if train else 1.0,
            self.dropouts["output"]: config.outputDropout if train else 1.0,
            self.dropouts["controlPre"]: config.controlPreDropout if train else 1.0,
            self.dropouts["controlPost"]: config.controlPostDropout if train else 1.0,
            self.dropouts["wordEmb"]: config.wordEmbDropout if train else 1.0,
            self.dropouts["word"]: config.wordDp if train else 1.0,
            self.dropouts["vocab"]: config.vocabDp if train else 1.0,
            self.dropouts["object"]: config.objectDp if train else 1.0,
            self.dropouts["wordStandard"]: config.wordStandardDp if train else 1.0,
            self.lr: config.lr,
            self.train: train
        }

        if config.imageObjects:
            feedDict.update({
                self.imagesAll: images["images"],
                self.imagesObjectNumAll: data["objectsNums"], 
            })                                   
        else:
            feedDict.update({
                self.imagesPlaceholder: images["images"]
            })

        if config.dataset == "VQA":
            feedDict.update({
                self.answerFreqListsAll: data["answerFreqs"],
                self.answerFreqNumsAll: data["answerFreqNums"]
            })

        if config.ansFormat == "mc":
            feedDict.update({
                self.choicesIndicesAll: data["choices"],
                self.choicesNumsAll: data["choicesNums"]
            })      

        return feedDict

    # Splits data to a specific GPU (tower) for parallelization
    def initTowerBatch(self, towerI, towersNum, dataSize):
        towerBatchSize = tf.floordiv(dataSize, towersNum)
        start = towerI * towerBatchSize
        end = (towerI + 1) * towerBatchSize if towerI < towersNum - 1 else dataSize

        self.questionIndices = self.questionIndicesAll[start:end]
        self.questionLengths = self.questionLengthsAll[start:end]
        
        self.images = self.imagesAll[start:end]

        self.imagesObjectNum = None
        if config.imageObjects:
            self.imagesObjectNum = self.imagesObjectNumAll[start:end]

        self.answerIndices = self.answerIndicesAll[start:end]

        self.answerFreqs = self.answerFreqNums = None
        if config.dataset == "VQA":
            self.answerFreqLists = self.answerFreqListsAll[start:end]
            self.answerFreqNums = self.answerFreqNumsAll[start:end]

        self.choicesIndices = self.choicesNums = None
        if config.ansFormat == "mc":
            self.choicesIndices = self.choicesIndicesAll[start:end]
            self.choicesNums = self.choicesNumsAll[start:end]

        self.batchSize = end - start

    '''
    The Image Input Unit (stem). Passes the image features through a CNN-network
    Optionally adds position encoding (doesn't in the default behavior).
    Flatten the image into Height * Width "Knowledge base" array.

    Args:
        images: image input. [batchSize, height, width, inDim]
        inDim: input image dimension
        outDim: image out dimension
        addLoc: if not None, adds positional encoding to the image

    Returns preprocessed images. 
    [batchSize, height * width, outDim]
    '''
    def stem(self, images, inDim, outDim, addLoc = None):
        with tf.variable_scope("stem"):        
            if config.stemNormalize:
                images = tf.nn.l2_normalize(images, dim = -1)

            if config.imageObjects: # VQA ??? or config.useBaseline:
                features, dim = images, inDim
                if config.stemLinear:
                    features = ops.linear(images, inDim, outDim, dropout = self.dropouts["stem"] if config.stemDp else 1.0)
                    dim = outDim
                elif config.stemDeep:
                    dims = [inDim] + config.stemDims + [outDim]
                    features = ops.FCLayer(features, dims, dropout = self.dropouts["stem"])                    

                if config.stemAct != "NON":
                    features = ops.actF(config.stemAct)(features)

                return features, dim

            if addLoc is None:
                addLoc = config.locationAware

            if config.stemLinear:
                features = ops.linear(images, inDim, outDim)
            else:
                if config.stemNumLayers == 0:
                    outDim = inDim
                else:
                    dims = [inDim] + ([config.stemDim] * (config.stemNumLayers - 1)) + [outDim]

                    if addLoc:
                        images, inDim = ops.addLocation(images, inDim, config.locationDim, 
                            h = self.H, w = self.W, locType = config.locationType) 
                        dims[0] = inDim

                    features = ops.CNNLayer(images, dims, 
                        batchNorm = self.batchNorm if config.stemBN else None,
                        dropout = self.dropouts["stem"],
                        kernelSizes = config.stemKernelSizes, 
                        strides = config.stemStrideSizes) 

                    if config.stemGridRnn:
                        features = ops.multigridRNNLayer(features, H, W, outDim)

            if config.baselineNew or (not config.useBaseline):
                features = tf.reshape(features, (self.batchSize, -1, outDim))
                
        return features, outDim      

    # Embed question using parametrized word embeddings.
    # The embedding are initialized to the values supported to the class initialization
    def qEmbeddingsOp(self, qIndices, embInit):
        with tf.variable_scope("qEmbeddings"):
            embInit = tf.to_float(embInit)
            embeddingsVar = tf.get_variable("emb", initializer = embInit, 
                dtype = tf.float32, trainable = (not config.wrdEmbQFixed))
            embeddings = tf.concat([tf.zeros((1, config.wrdQEmbDim)), embeddingsVar], axis = 0)
            questions = tf.nn.embedding_lookup(embeddings, qIndices)

        return questions, embeddings

    # Embed answer words
    def aEmbeddingsOp(self, aIndices, embInit):
        with tf.variable_scope("aEmbeddings"):
            if embInit is None:
                return None
            embInit = tf.to_float(embInit)
            embeddings = tf.get_variable("emb", initializer = embInit, 
                dtype = tf.float32, trainable = (not config.wrdEmbAFixed))

            if config.ansFormat == "mc":
                answers = tf.nn.embedding_lookup(embeddings, aIndices)
            else:
                answers = embeddings
        return answers

    def vocabEmbeddings(self, embInit, name):
        with tf.variable_scope("vocabEmbeddings" + name):
            embInit = tf.to_float(embInit)
            embeddings = tf.get_variable("emb", initializer = embInit, 
                dtype = tf.float32, trainable = (not config.semanticFixEmbs))
        return embeddings

    # Embed question and answer words with tied embeddings
    def qaEmbeddingsOp(self, qIndices, aIndices, embInit):
        questions, embeddings = self.qEmbeddingsOp(qIndices, embInit)    
        answers = tf.nn.embedding_lookup(embeddings, aIndices)
        return questions, answers, embeddings

    '''
    Embed question (and optionally answer) using parametrized word embeddings.
    The embedding are initialized to the values supported to the class initialization 
    '''
    def embeddingsOp(self, qIndices, aIndices, embInit):
        # nullWord = tf.tile(tf.expand_dims(nullWord, axis = 0), [self.batchSize, 1, 1])
        if config.ansEmbMod == "SHARED":
            if config.ansFormat == "oe":
            #if aIndices is None:
                aIndices = embInit["oeAnswers"]
            questions, answers, qaEmbeddings = self.qaEmbeddingsOp(qIndices, aIndices, embInit["qa"])
        else:
            questions, qEmbeddings = self.qEmbeddingsOp(qIndices, embInit["q"])
            answers = self.aEmbeddingsOp(aIndices, embInit["a"])

        if config.ansFormat == "oe" and config.ansEmbMod != "NON":
            answers = tf.tile(tf.expand_dims(answers, axis = 0), [self.batchSize, 1, 1])

        return questions, answers # , embeddings

    '''
    The Question Input Unit embeds the questions to randomly-initialized word vectors,
    and runs a recurrent bidirectional encoder (RNN/LSTM etc.) that gives back
    vector representations for each question (the RNN final hidden state), and
    representations for each of the question words (the RNN outputs for each word). 

    The method uses bidirectional LSTM, by default.
    Optionally projects the outputs of the LSTM (with linear projection / 
    optionally with some activation).
    
    Args:
        questions: question word embeddings  
        [batchSize, questionLength, wordEmbDim]

        questionLengths: the question lengths.
        [batchSize]

        projWords: True to apply projection on RNN outputs.
        projQuestion: True to apply projection on final RNN state.
        projDim: projection dimension in case projection is applied.  

    Returns:
        Contextual Words: RNN outputs for the words.
        [batchSize, questionLength, ctrlDim]

        Vectorized Question: Final hidden state representing the whole question.
        [batchSize, ctrlDim]
    '''
    def encoder(self, questions, questionLengths, projWords = False, 
        projQuestion = False, projDim = None):
        
        with tf.variable_scope("encoder"):
            # variational dropout option
            varDp = None
            if config.encVariationalDropout:
                varDp = {"stateDp": self.dropouts["stateInput"], 
                         "inputDp": self.dropouts["encInput"], 
                         "inputSize": config.wrdQEmbDim}

            # rnns
            for i in range(config.encNumLayers):
                questionCntxWords, vecQuestions = ops.RNNLayer(questions, questionLengths, 
                    config.encDim, bi = config.encBi, cellType = config.encType, 
                    dropout = self.dropouts["encInput"], varDp = varDp, name = "rnn%d" % i)

            # dropout for the question vector
            vecQuestions = tf.nn.dropout(vecQuestions, self.dropouts["question"])
            
            # projection of encoder outputs 
            if projWords:
                questionCntxWords = ops.linear(questionCntxWords, config.encDim, projDim, 
                    name = "projCW")
            if projQuestion:
                vecQuestions = ops.linear(vecQuestions, config.encDim, projDim, 
                    act = config.encProjQAct, name = "projQ")

        return questionCntxWords, vecQuestions        

    '''
    Stacked Attention Layer for baseline. Computes interaction between images
    and the previous memory, and casts it back to compute attention over the 
    image, which in turn is summed up with the previous memory to result in the
    new one. 

    Args:
        images: input image.
        [batchSize, H * W, inDim]

        memory: previous memory value
        [batchSize, inDim]

        inDim: inputs dimension
        hDim: hidden dimension to compute interactions between image and memory

    Returns the new memory value.
    '''
    def baselineAttLayer(self, images, memory, inDim, hDim, name = "", reuse = None):
        with tf.variable_scope("attLayer" + name, reuse = reuse):         
            # projImages = ops.linear(images, inDim, hDim, name = "projImage")
            # projMemory = tf.expand_dims(ops.linear(memory, inDim, hDim, name = "projMemory"), axis = -2)       
            # if config.saMultiplicative:
            #     interactions = projImages * projMemory
            # else:
            #     interactions = tf.tanh(projImages + projMemory) 
            interactions, hDim = ops.mul(images, memory, inDim, proj = {"dim": hDim, "shared": False}, 
                interMod = config.baselineAttType)
            
            attention = ops.inter2att(interactions, hDim, mask = self.imagesObjectNum)
            summary = ops.att2Smry(attention, images)            

            newMemory = memory + summary
        
        return newMemory


    '''
    Baseline approach:
    If baselineAtt is True, applies several layers (baselineAttNumLayers)
    of stacked attention to image and memory, when memory is initialized
    to the vector questions. See baselineAttLayer for further details.

    Otherwise, computes result output features based on image representation
    (baselineCNN), or question (baselineLSTM) or both.

    Args:
        vecQuestions: question vector representation
        [batchSize, questionDim]

        questionDim: dimension of question vectors

        images: (flattened) image representation
        [batchSize, imageDim]

        imageDim: dimension of image representations.
        
        hDim: hidden dimension to compute interactions between image and memory
        (for attention-based baseline).

    Returns final features to use in later classifier.
    [batchSize, outDim] (out dimension depends on baseline method)
    '''
    def baseline(self, vecQuestions, questionDim, images, imageDim, hDim):
        with tf.variable_scope("baseline"):
            if config.baselineAtt:                
                memory = ops.linear(vecQuestions, questionDim, hDim, name = "qProj")
                images = ops.linear(images, imageDim, hDim, name = "iProj")

                for i in range(config.baselineAttNumLayers):
                    memory = self.baselineAttLayer(images, memory, hDim, hDim, 
                        name = "baseline%d" % i)
                memDim = hDim
            else:      
                if config.imageObjects:
                    cff = tf.get_variable("cff", shape = (imageDim, ), initializer = tf.random_normal_initializer())                     
                    interactions, hDim = ops.mul(images, cff, imageDim)
                    attention = ops.inter2att(interactions, hDim, mask = self.imagesObjectNum)
                    images = ops.att2Smry(attention, images)
                else:
                    images, imageDim = ops.linearizeFeatures(images, self.H, self.W, 
                        imageDim, projDim = config.baselineProjDim)
                if config.baselineLSTM and config.baselineCNN:
                    memory = tf.concat([vecQuestions, images], axis = -1)
                    memDim = questionDim + imageDim
                elif config.baselineLSTM:
                    memory = vecQuestions
                    memDim = questionDim
                else: # config.baselineCNN
                    memory = images
                    memDim = imageDim 
                
        return memory, memDim

    '''
    Runs the MAC recurrent network to perform the reasoning process.
    Initializes a MAC cell and runs netLength iterations.
    
    Currently it passes the question and knowledge base to the cell during
    its creating, such that it doesn't need to interact with it through 
    inputs / outputs while running. The recurrent computation happens 
    by working iteratively over the hidden (control, memory) states.  
    
    Args:
        images: flattened image features. Used as the "Knowledge Base".
        (Received by default model behavior from the Image Input Units).
        [batchSize, H * W, memDim]

        vecQuestions: vector questions representations.
        (Received by default model behavior from the Question Input Units
        as the final RNN state).
        [batchSize, ctrlDim]

        questionWords: question word embeddings.
        [batchSize, questionLength, ctrlDim]

        questionCntxWords: question contextual words.
        (Received by default model behavior from the Question Input Units
        as the series of RNN output states).
        [batchSize, questionLength, ctrlDim]

        questionLengths: question lengths.
        [batchSize]

    Returns the final control state and memory state resulted from the network.
    ([batchSize, ctrlDim], [bathSize, memDim])
    '''
    def MACnetwork(self, images, vecQuestions, questionWords, questionCntxWords, 
        questionLengths, name = "", reuse = None):

        with tf.variable_scope("MACnetwork" + name, reuse = reuse):
            
            self.macCell = MACCell(
                vecQuestions = vecQuestions,
                questionWords = questionWords,
                questionCntxWords = questionCntxWords, 
                questionLengths = questionLengths,
                knowledgeBase = images,
                kbSize = self.imagesObjectNum, 
                memoryDropout = self.dropouts["memory"],
                readDropout = self.dropouts["read"],
                writeDropout = self.dropouts["write"],
                controlDropoutPre = self.dropouts["controlPre"],
                controlDropoutPost = self.dropouts["controlPost"],
                wordDropout = self.dropouts["word"],
                vocabDropout = self.dropouts["vocab"],
                objectDropout = self.dropouts["object"],
                # qDropoutMAC = self.qDropoutMAC,
                batchSize = self.batchSize,
                train = self.train,
                reuse = reuse)           

            state = self.macCell.zero_state(self.batchSize, tf.float32)

            none = tf.zeros((self.batchSize, 1), dtype = tf.float32)

            for i in range(config.netLength):
                self.macCell.iteration = i
                _, state = self.macCell(none, state)                     

            finalControl = state.control
            finalMemory = state.memory

        return finalControl, finalMemory         

    '''
    Output Unit (step 1): chooses the inputs to the output classifier.
    
    By default the classifier input will be the the final memory state of the MAC network.
    If outQuestion is True, concatenate the question representation to that.
    If outImage is True, concatenate the image flattened representation.

    Args:
        memory: (final) memory state of the MAC network.
        [batchSize, memDim]

        vecQuestions: question vector representation.
        [batchSize, ctrlDim]

        images: image features.
        [batchSize, H, W, imageInDim]

        imageInDim: images dimension.

    Returns the resulted features and their dimension. 
    '''
    def outputOp(self, memory, control, vecQuestions, images, imageInDim):
        with tf.variable_scope("outputUnit"):            
            features = memory
            dim = config.memDim

            if config.outQuestion:
                q = vecQuestions
                eQ = ops.linear(q, config.ctrlDim, config.memDim, name = "outQuestion") 
                features, dim = ops.concat(features, eQ, config.memDim, mul = config.outQuestionMul)
            
            # assumes imageObjects False
            if config.outImage:
                images, imagesDim = ops.linearizeFeatures(images, self.H, self.W, self.imageInDim, 
                    outputDim = config.outImageDim)
                images = ops.linear(images, config.memDim, config.outImageDim, name = "outImage")
                features = tf.concat([features, images], axis = -1)
                dim += config.outImageDim

        return features, dim        

    '''
    Output Unit (step 2): Computes the logits for the answers. Passes the features
    through fully-connected network to get the logits over the possible answers.
    Optionally uses answer word embeddings in computing the logits (by default, it doesn't).

    Args:
        features: features used to compute logits
        [batchSize, inDim]

        inDim: features dimension

        aEmbedding: supported word embeddings for answer words in case answerMod is not NON.
        Optionally computes logits by computing dot-product with answer embeddings.
    
    Returns: the computed logits.
    [batchSize, answerWordsNum]
    '''
    # in mc has to be ansMod not NON
    def classifier(self, features, inDim, choices = None, choicesNums = None):
        with tf.variable_scope("classifier"):                    
            outDim = config.answerWordsNum
            dims = [inDim] + config.outClassifierDims + [outDim]
            if config.answerMod != "NON":
                dims[-1] = config.wrdAEmbDim                

            logits = ops.FCLayer(features, dims, 
                batchNorm = self.batchNorm if config.outputBN else None, 
                dropout = self.dropouts["output"]) 
            
            if config.answerMod != "NON":
                logits = ops.gatedAct(config.outAct, gate = config.outGate)(logits)
                logits = tf.nn.dropout(logits, self.dropouts["output"])
                concat = {"x": config.answerBias}
                interactions, interDim = ops.mul(choices, logits, dims[-1], interMod = config.answerMod, concat = concat)
                logits = ops.inter2logits(interactions, interDim, sumMod = config.answerSumMod)
                if config.ansFormat == "oe":
                    logits += ops.getBias((outDim, ), "ans")
                else:
                    logits = ops.expMask(logits, choicesNums)

        return logits

    def aggregateFreqs(self, answerFreqs, answerFreqNums):
        if answerFreqs is None:
            return None
        answerFreqs = tf.one_hot(answerFreqs, config.answerWordsNum) # , axis = -1
        mask = tf.sequence_mask(answerFreqNums, maxlen = config.AnswerFreqMaxNum)
        mask = tf.expand_dims(tf.to_float(mask), axis = -1)
        answerFreqs *= mask
        answerFreqs = tf.reduce_sum(answerFreqs, axis = 1)
        return answerFreqs

    # Computes mean cross entropy loss between logits and answers.
    def addAnswerLossOp(self, logits, answers, answerFreqs, answerFreqNums):
        if config.lossType == "softmax": # or config.ansFormat == "mc":
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = answers, logits = logits)
        
        elif config.lossType == "svm":
            answers = tf.one_hot(answers, config.answerWordsNum) # , axis = -1
            losses = ops.hingeLoss(labels = answers, logits = logits)
        
        elif config.lossType == "probSoftmax":
            answerFreqs = tf.to_float(answerFreqs)
            answerDist = answerFreqs / tf.expand_dims(tf.to_float(answerFreqNums), axis = -1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels = answerDist, logits = logits)
            if config.weightedSoftmax:
                weights = tf.to_float(answerFreqNums) / float(config.AnswerFreqMaxNum)
                losses *= weights
        elif config.lossType == "sigmoid":
            if config.dataset == "VQA":
                answerFreqs = tf.to_float(answerFreqs)
                answerDist = answerFreqs / float(config.AnswerFreqMaxNum)
            else:
                answerDist = tf.one_hot(answers, config.answerWordsNum)
            if config.lossWeight == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = answerDist, logits = logits)
            else:
                print("weighted sigmoid")
                losses = tf.nn.weighted_cross_entropy_with_logits(targets = answerDist, logits = logits, 
                    pos_weight = config.lossWeight)
            if config.ansWeighting or config.ansWeightingRoot:
                losses *= self.answerDict.weights
            losses = tf.reduce_sum(losses, axis = -1)
        else:
            print("non-identified loss")
        loss = tf.reduce_mean(losses)
        self.answerLossList.append(loss)

        return loss, losses

    # Computes predictions (by finding maximal logit value, corresponding to highest probability)
    # and mean accuracy between predictions and answers. 
    def addPredOp(self, logits, answers): # , answerFreqs
        with tf.variable_scope("pred"):
            if config.ansFormat == "oe":# and config.ansAddUnk:
                mask = tf.to_float(tf.sequence_mask([2], config.answerWordsNum)) * (-1e30) # 1 or 2?
                logits += mask
            
            preds = tf.to_int32(tf.argmax(logits, axis = -1)) # tf.nn.softmax( 

            if config.dataset == "VQA" and config.ansFormat == "oe":
                agreeing = tf.reduce_sum(tf.one_hot(preds, config.answerWordsNum) * self.answerFreqs, axis = -1)
                corrects = tf.minimum(agreeing * 0.3, 1.0) # /3 ?
            else:
                corrects = tf.to_float(tf.equal(preds, answers)) 

            correctNum = tf.reduce_sum(corrects)
            acc = tf.reduce_mean(corrects)
            self.correctNumList.append(correctNum) 
            self.answerAccList.append(acc)

        return preds, corrects, correctNum
    
    # Creates optimizer (adam)
    def addOptimizerOp(self): 
        with tf.variable_scope("trainAddOptimizer"):            
            self.globalStep = tf.Variable(0, dtype = tf.int32, trainable = False, name = "globalStep") # init to 0 every run?
            optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
            if config.subsetOpt:
                self.subsetOptimizer = tf.train.AdamOptimizer(learning_rate = self.lr * config.subsetOptMult)

        return optimizer

    '''
    Computes gradients for all variables or subset of them, based on provided loss, 
    using optimizer.
    '''
    def computeGradients(self, optimizer, loss, trainableVars = None): # tf.trainable_variables()
        with tf.variable_scope("computeGradients"):            
            if config.trainSubset:
                trainableVars = []
                allVars = tf.trainable_variables()
                for var in allVars:
                    if any((s in var.name) for s in config.varSubset):
                        trainableVars.append(var)

            if config.subsetOpt:
                trainableVars = []
                subsetVars = []
                allVars = tf.trainable_variables()                
                for var in allVars:
                    if any((s in var.name) for s in config.varSubset):
                        subsetVars.append(var)
                    else:
                        trainableVars.append(var)

            gradients_vars = optimizer.compute_gradients(loss, trainableVars)

            if config.subsetOpt:
                self.subset_gradients_vars = self.subsetOptimizer.compute_gradients(loss, subsetVars) 
                self.subset_gradientVarsList.append(self.subset_gradients_vars)

        return gradients_vars

    '''
    Apply gradients. Optionally clip them, and update exponential moving averages 
    for parameters.
    '''
    def addTrainingOp(self, optimizer, gradients_vars):
        with tf.variable_scope("train"):
            gradients, variables = zip(*gradients_vars)
            norm = tf.global_norm(gradients)

            # gradient clipping
            if config.clipGradients:            
                clippedGradients, _ = tf.clip_by_global_norm(gradients, config.gradMaxNorm, use_norm = norm)
                gradients_vars = zip(clippedGradients, variables)

            # updates ops (for batch norm) and train op
            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updateOps):
                train = optimizer.apply_gradients(gradients_vars, global_step = self.globalStep)

                if config.subsetOpt:
                    subsetTrain = self.subsetOptimizer.apply_gradients(self.subset_gradientVarsAll)
                    train = tf.group(train, subsetTrain)

            # exponential moving average
            if config.useEMA:
                ema = tf.train.ExponentialMovingAverage(decay = config.emaDecayRate)
                maintainAveragesOp = ema.apply(tf.trainable_variables())

                with tf.control_dependencies([train]):
                    trainAndUpdateOp = tf.group(maintainAveragesOp)
                
                train = trainAndUpdateOp

                self.emaDict = ema.variables_to_restore()

        return train, norm

    def averageAcrossTowers(self, gpusNum):        
        if gpusNum == 1:
            self.lossAll = self.lossList[0]
            self.answerLossAll = self.answerLossList[0]
            self.answerAccAll = self.answerAccList[0]
            self.correctNumAll = self.correctNumList[0]
            self.predsAll = self.predsList[0]
            self.gradientVarsAll = self.gradientVarsList[0]
            if config.subsetOpt:
                self.subset_gradientVarsAll = self.subset_gradientVarsList[0]
        else:
            self.lossAll = tf.reduce_mean(tf.stack(self.lossList, axis = 0), axis = 0)
            self.answerLossAll = tf.reduce_mean(tf.stack(self.answerLossList, axis = 0), axis = 0)
            self.answerAccAll = tf.reduce_mean(tf.stack(self.answerAccList, axis = 0), axis = 0)
            self.correctNumAll = tf.reduce_sum(tf.stack(self.correctNumList, axis = 0), axis = 0)
            self.predsAll = tf.concat(self.predsList, axis = 0)

            self.gradientVarsAll = []
            for grads_var in zip(*self.gradientVarsList):
                gradients, variables = zip(*grads_var)
                if gradients[0] != None:
                    avgGradient = tf.reduce_mean(tf.stack(gradients, axis = 0), axis = 0)
                else:
                    avgGradient = None                
                var = variables[0]
                grad_var = (avgGradient, var)
                self.gradientVarsAll.append(grad_var)

            if config.subsetOpt:
                self.subset_gradientVarsAll = []
                for grads_var in zip(*self.subset_gradientVarsList):
                    gradients, variables = zip(*grads_var)
                    if gradients[0] != None:
                        avgGradient = tf.reduce_mean(tf.stack(gradients, axis = 0), axis = 0)
                    else:
                        avgGradient = None                
                    var = variables[0]
                    grad_var = (avgGradient, var)
                    self.subset_gradientVarsAll.append(grad_var)

    def trim2DVectors(self, vectors, vectorsLengths):
        maxLength = np.max(vectorsLengths)
        return vectors[:,:maxLength]

    def trimData(self, data):
        data["questions"] = self.trim2DVectors(data["questions"], data["questionLengths"])
        return data

    '''
    Builds predictions JSON, by adding the model's predictions and attention maps 
    back to the original data JSON.
    '''
    def buildPredsList(self, data, predictions, attentionMaps):
        predsList = []
        
        for i, instance in enumerate(data["instances"]):

            if predictions is not None:
                if config.ansFormat == "oe":
                    pred = self.answerDict.decodeId(predictions[i])
                else:
                    pred = instance["choices"][predictions[i]]
                instance["prediction"] = pred          

            # aggregate np attentions of instance i in the batch into 2d list
            attMapToList = lambda attMap: [step[i].tolist() for step in attMap]
            if attentionMaps is not None:
                attentions = {k: attMapToList(attentionMaps[k]) for k in attentionMaps}
                instance["attentions"] = attentions

            predsList.append(instance)

        return predsList

    '''
    Processes a batch of data with the model.

    Args:
        sess: TF session
        
        data: Data batch. Dictionary that contains numpy array for:
        questions, questionLengths, answers. 
        See preprocess.py for further information of the batch structure.

        images: batch of image features, as numpy array. images["images"] contains
        [batchSize, channels, h, w]

        train: True to run batch for training.

        getAtt: True to return attention maps for question and image (and optionally 
        self-attention and gate values).

    Returns results: e.g. loss, accuracy, running time.
    '''
    def runBatch(self, sess, data, images, train, getPreds = False, getAtt = False, allData = None):     
        batchSizeOp = self.batchSizeAll
        indicesOp = self.noOp
        
        trainOp = self.trainOp if train else self.noOp
        gradNormOp = self.gradNorm if train else self.noOp

        predsOp = (self.predsAll, self.correctNumAll, self.answerAccAll)
        
        attOp = self.macCell.attentions if not config.useBaseline else (self.attentions if config.baselineNew else self.noOp)
        
        time0 = time.time()
        feed = self.createFeedDict(data, images, train) 

        time1 = time.time()
        batchSize, indices, _, loss, predsInfo, gradNorm, attentionMaps = sess.run(
            [batchSizeOp, indicesOp, trainOp, self.lossAll, predsOp, gradNormOp, attOp], 
            feed_dict = feed)
        
        time2 = time.time()  
        
        predsList = []
        if getPreds:
            if data is None:
                data = [allData["instances"][i] for i in indices]
            predsList = self.buildPredsList(data, predsInfo[0], attentionMaps if getAtt else None)

        return {"loss": loss,
                "correctNum": predsInfo[1],
                "acc": predsInfo[2], 
                "preds": predsList,
                "gradNorm": gradNorm if train else -1,
                "readTime": time1 - time0,
                "trainTime": time2 - time1,
                "batchSize": batchSize}

    def build(self):
        self.addPlaceholders()
        self.optimizer = self.addOptimizerOp()

        self.gradientVarsList = []
        if config.subsetOpt:
            self.subset_gradientVarsList = []
        self.lossList = []

        self.answerLossList = []
        self.correctNumList = []
        self.answerAccList = []
        self.predsList = []

        with tf.variable_scope("macModel"):
            for i in range(config.gpusNum):
                with tf.device("/gpu:{}".format(i)):
                    with tf.name_scope("tower{}".format(i)) as scope:
                        self.initTowerBatch(i, config.gpusNum, self.batchSizeAll)

                        self.loss = tf.constant(0.0)

                        # embed questions words (and optionally answer words)
                        questionWords, choices = self.embeddingsOp(self.questionIndices, 
                            self.choicesIndices, self.embeddingsInit)

                        projWords = projQuestion = ((config.encDim != config.ctrlDim) or config.encProj)
                        questionCntxWords, vecQuestions = self.encoder(questionWords, 
                            self.questionLengths, projWords, projQuestion, config.ctrlDim)

                        # Image Input Unit (stem)
                        imageFeatures, imageDim = self.stem(self.images, self.imageInDim, config.memDim)

                        # baseline model
                        if config.useBaseline:
                            # inpImg = imageFeatures if config.baselineNew else self.images
                            # inpDim = imageDim if config.baselineNew else self.imageInDim
                            output, dim = self.baseline(vecQuestions, config.ctrlDim, 
                                imageFeatures, imageDim, config.attDim) # self.images
                        # MAC model
                        else:                                  
                            finalControl, finalMemory = self.MACnetwork(imageFeatures, vecQuestions, 
                                questionWords, questionCntxWords, self.questionLengths)
                            
                            # Output Unit - step 1 (preparing classifier inputs)
                            output, dim = self.outputOp(finalMemory, finalControl, vecQuestions, 
                                self.images, self.imageInDim)

                        # Output Unit - step 2 (classifier)
                        logits = self.classifier(output, dim, choices, self.choicesNums)

                        # compute loss, predictions, accuracy
                        if config.dataset == "VQA":
                            self.answerFreqs = self.aggregateFreqs(self.answerFreqLists, self.answerFreqNums)
                        else:
                            self.answerFreqs = None
                            self.answerFreqNums = None
                        answerLoss, self.losses = self.addAnswerLossOp(logits, self.answerIndices, 
                            self.answerFreqs, self.answerFreqNums)
                        self.preds, self.corrects, self.correctNum = self.addPredOp(logits, self.answerIndices) # ,self.answerFreqs

                        self.loss += answerLoss
                        self.predsList.append(self.preds)
                        self.lossList.append(self.loss)

                        # compute gradients
                        gradient_vars = self.computeGradients(self.optimizer, self.loss, trainableVars = None)
                        self.gradientVarsList.append(gradient_vars)

                        # reuse variables in next towers
                        tf.get_variable_scope().reuse_variables()

        self.averageAcrossTowers(config.gpusNum)

        self.trainOp, self.gradNorm = self.addTrainingOp(self.optimizer, self.gradientVarsAll)
        self.noOp = tf.no_op()
