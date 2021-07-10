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
    def __init__(self, embeddingsInit, answerDict):
        self.embeddingsInit = embeddingsInit
        self.answerDict = answerDict
        self.build()

    '''
    Initializes placeholders.
        questionsIndicesAll: integer ids of question words. 
        [batchSize, questionLength]
        
        questionLengthsAll: length of each question. 
        [batchSize]
        
        imagesPlaceholder: image features. 
        [batchSize, channels, height, width]
        (converted internally to [batchSize, height, width, channels])
        
        answersIndicesAll: integer ids of answer words. 
        [batchSize]

        lr: learning rate (tensor scalar)
        train: train / evaluation (tensor boolean)

        dropout values dictionary (tensor scalars)
    '''
    # change to H x W x C?
    def addPlaceholders(self):
        with tf.compat.v1.variable_scope("Placeholders"):
            ## data
            # questions            
            self.questionsIndicesAll = tf.compat.v1.placeholder(tf.int32, shape = (None, None))
            self.questionLengthsAll = tf.compat.v1.placeholder(tf.int32, shape = (None, ))

            # images
            # put image known dimension as last dim?
            self.imagesPlaceholder = tf.compat.v1.placeholder(tf.float32, shape = (None, None, None, None))
            self.imagesAll = tf.transpose(self.imagesPlaceholder, (0, 2, 3, 1))
            # self.imageH = tf.shape(self.imagesAll)[1]
            # self.imageW = tf.shape(self.imagesAll)[2]

            # answers
            self.answersIndicesAll = tf.compat.v1.placeholder(tf.int32, shape = (None, ))

            ## optimization
            self.lr = tf.compat.v1.placeholder(tf.float32, shape = ())
            self.train = tf.compat.v1.placeholder(tf.bool, shape = ())
            self.batchSizeAll = tf.shape(self.questionsIndicesAll)[0]

            ## dropouts
            # TODO: change dropouts to be 1 - current
            self.dropouts = {
                "encInput": tf.compat.v1.placeholder(tf.float32, shape = ()),
                "encState": tf.compat.v1.placeholder(tf.float32, shape = ()),
                "stem": tf.compat.v1.placeholder(tf.float32, shape = ()),
                "question": tf.compat.v1.placeholder(tf.float32, shape = ()),
                # self.dropouts["question"]Out = tf.compat.v1.placeholder(tf.float32, shape = ())
                # self.dropouts["question"]MAC = tf.compat.v1.placeholder(tf.float32, shape = ())
                "read": tf.compat.v1.placeholder(tf.float32, shape = ()),
                "write": tf.compat.v1.placeholder(tf.float32, shape = ()),
                "memory": tf.compat.v1.placeholder(tf.float32, shape = ()),
                "output": tf.compat.v1.placeholder(tf.float32, shape = ())
            }

            # batch norm params
            self.batchNorm = {"decay": config.bnDecay, "train": self.train}

            # if config.parametricDropout:
            #     self.dropouts["question"] = parametricDropout("qDropout", self.train)
            #     self.dropouts["read"] = parametricDropout("readDropout", self.train)
            # else:
            #     self.dropouts["question"] = self.dropouts["_q"]
            #     self.dropouts["read"] = self.dropouts["_read"]
            
            # if config.tempDynamic:
            #     self.tempAnnealRate = tf.compat.v1.placeholder(tf.float32, shape = ())

            self.H, self.W, self.imageInDim = config.imageDims

    # Feeds data into placeholders. See addPlaceholders method for further details.
    def createFeedDict(self, data, images, train):
        feedDict = {
            self.questionsIndicesAll: data["questions"],
            self.questionLengthsAll: data["questionLengths"],
            self.imagesPlaceholder: images["images"],
            self.answersIndicesAll: data["answers"],
            
            self.dropouts["encInput"]: config.encInputDropout if train else 1.0,
            self.dropouts["encState"]: config.encStateDropout if train else 1.0,
            self.dropouts["stem"]: config.stemDropout if train else 1.0,
            self.dropouts["question"]: config.qDropout if train else 1.0, #_
            self.dropouts["memory"]: config.memoryDropout if train else 1.0,
            self.dropouts["read"]: config.readDropout if train else 1.0, #_
            self.dropouts["write"]: config.writeDropout if train else 1.0,
            self.dropouts["output"]: config.outputDropout if train else 1.0,
            # self.dropouts["question"]Out: config.qDropoutOut if train else 1.0,
            # self.dropouts["question"]MAC: config.qDropoutMAC if train else 1.0,

            self.lr: config.lr,
            self.train: train
        }

        # if config.tempDynamic:
        #     feedDict[self.tempAnnealRate] = tempAnnealRate          

        return feedDict

    # Splits data to a specific GPU (tower) for parallelization
    def initTowerBatch(self, towerI, towersNum, dataSize):
        towerBatchSize = tf.compat.v1.floordiv(dataSize, towersNum)
        start = towerI * towerBatchSize
        end = (towerI + 1) * towerBatchSize if towerI < towersNum - 1 else dataSize

        self.questionsIndices = self.questionsIndicesAll[start:end]
        self.questionLengths = self.questionLengthsAll[start:end]
        self.images = self.imagesAll[start:end]
        self.answersIndices = self.answersIndicesAll[start:end]

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

        with tf.compat.v1.variable_scope("stem"):
            if addLoc is None:
                addLoc = config.locationAware

            if config.stemLinear:
                features = ops.linear(images, inDim, outDim)
            else:
                dims = [inDim] + ([config.stemDim] * (config.stemNumLayers - 1)) + [outDim]

                if addLoc:
                    images, inDim = ops.addLocation(images, inDim, config.locationDim, 
                        h = self.H, w = self.W, locType = config.locationType) 
                    dims[0] = inDim

                    # if config.locationType == "PE":
                    #     dims[-1] /= 4
                    #     dims[-1] *= 3
                    # else:
                    #     dims[-1] -= 2
                features = ops.CNNLayer(images, dims, 
                    batchNorm = self.batchNorm if config.stemBN else None,
                    dropout = self.dropouts["stem"],
                    kernelSizes = config.stemKernelSizes, 
                    strides = config.stemStrideSizes)

                # if addLoc:
                #     lDim = outDim / 4
                #     lDim /= 4
                #     features, _ = addLocation(features, dims[-1], lDim, h = H, w = W, 
                #         locType = config.locationType) 

                if config.stemGridRnn:
                    features = ops.multigridRNNLayer(features, H, W, outDim)

            # flatten the 2d images into a 1d KB
            features = tf.reshape(features, (self.batchSize, -1, outDim))

        return features  

    # Embed question using parametrized word embeddings.
    # The embedding are initialized to the values supported to the class initialization
    def qEmbeddingsOp(self, qIndices, embInit):
        with tf.compat.v1.variable_scope("qEmbeddings"):
            # if config.useCPU:
            #     with tf.device('/cpu:0'):
            #         embeddingsVar = tf.Variable(self.embeddingsInit, name = "embeddings", dtype = tf.float32)
            # else:
            #     embeddingsVar = tf.Variable(self.embeddingsInit, name = "embeddings", dtype = tf.float32)
            embeddingsVar = tf.compat.v1.get_variable("emb", initializer = tf.compat.v1.to_float(embInit),
                dtype = tf.float32, trainable = (not config.wrdEmbFixed))
            embeddings = tf.concat([tf.zeros((1, config.wrdEmbDim)), embeddingsVar], axis = 0)
            questions = tf.nn.embedding_lookup(embeddings, qIndices)

        return questions, embeddings

    # Embed answer words
    def aEmbeddingsOp(self, embInit):
        with tf.compat.v1.variable_scope("aEmbeddings"):
            if embInit is None:
                return None
            answerEmbeddings = tf.compat.v1.get_variable("emb", initializer = tf.compat.v1.to_float(embInit),
                dtype = tf.float32)
        return answerEmbeddings

    # Embed question and answer words with tied embeddings
    def qaEmbeddingsOp(self, qIndices, embInit):
        questions, qaEmbeddings = self.qEmbeddingsOp(qIndices, embInit["qa"])
        aEmbeddings = tf.nn.embedding_lookup(qaEmbeddings, embInit["ansMap"])

        return questions, qaEmbeddings, aEmbeddings 

    '''
    Embed question (and optionally answer) using parametrized word embeddings.
    The embedding are initialized to the values supported to the class initialization 
    '''
    def embeddingsOp(self, qIndices, embInit):
        if config.ansEmbMod == "SHARED":
            questions, qEmb, aEmb = self.qaEmbeddingsOp(qIndices, embInit)                                
        else:
            questions, qEmb = self.qEmbeddingsOp(qIndices, embInit["q"])
            aEmb = self.aEmbeddingsOp(embInit["a"])

        return questions, qEmb, aEmb

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
        
        with tf.compat.v1.variable_scope("encoder"):
            # variational dropout option
            varDp = None
            if config.encVariationalDropout:
                varDp = {"stateDp": self.dropouts["stateInput"], 
                         "inputDp": self.dropouts["encInput"], 
                         "inputSize": config.wrdEmbDim}

            # rnns
            for i in range(config.encNumLayers):
                questionCntxWords, vecQuestions = ops.RNNLayer(questions, questionLengths, 
                    config.encDim, bi = config.encBi, cellType = config.encType, 
                    dropout = self.dropouts["encInput"], varDp = varDp, name = "rnn%d" % i)

            # dropout for the question vector
            vecQuestions = tf.compat.v1.nn.dropout(vecQuestions, self.dropouts["question"])
            
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
        with tf.compat.v1.variable_scope("attLayer" + name, reuse = reuse):
            # projImages = ops.linear(images, inDim, hDim, name = "projImage")
            # projMemory = tf.expand_dims(ops.linear(memory, inDim, hDim, name = "projMemory"), axis = -2)       
            # if config.saMultiplicative:
            #     interactions = projImages * projMemory
            # else:
            #     interactions = tf.tanh(projImages + projMemory) 
            interactions, _ = ops.mul(images, memory, inDim, proj = {"dim": hDim, "shared": False}, 
                interMod = config.baselineAttType)
            
            attention = ops.inter2att(interactions, hDim)
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
        with tf.compat.v1.variable_scope("baseline"):
            if config.baselineAtt:  
                memory = self.linear(vecQuestions, questionDim, hDim, name = "qProj")
                images = self.linear(images, imageDim, hDim, name = "iProj")

                for i in range(config.baselineAttNumLayers):
                    memory = self.baselineAttLayer(images, memory, hDim, hDim, 
                        name = "baseline%d" % i)
                memDim = hDim
            else:      
                images, imagesDim = ops.linearizeFeatures(images, self.H, self.W, 
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

        with tf.compat.v1.variable_scope("MACnetwork" + name, reuse = reuse):
            
            self.macCell = MACCell(
                vecQuestions = vecQuestions,
                questionWords = questionWords,
                questionCntxWords = questionCntxWords, 
                questionLengths = questionLengths,
                knowledgeBase = images,
                memoryDropout = self.dropouts["memory"],
                readDropout = self.dropouts["read"],
                writeDropout = self.dropouts["write"],
                # qDropoutMAC = self.qDropoutMAC,
                batchSize = self.batchSize,
                train = self.train,
                reuse = reuse)           

            state = self.macCell.zero_state(self.batchSize, tf.float32)

            # inSeq = tf.unstack(inSeq, axis = 1)        
            none = tf.zeros((self.batchSize, 1), dtype = tf.float32)

            # for i, inp in enumerate(inSeq):
            for i in range(config.netLength):
                self.macCell.iteration = i
                # if config.unsharedCells:
                    # with tf.compat.v1.variable_scope("iteration%d" % i):
                    # macCell.myNameScope = "iteration%d" % i
                _, state = self.macCell(none, state)                     
                # else:
                    # _, state = macCell(none, state)
                    # macCell.reuse = True

            # self.autoEncMMLoss = macCell.autoEncMMLossI
            # inputSeqL = None
            # _, lastOutputs = tf.nn.dynamic_rnn(macCell, inputSeq, # / static
            #     sequence_length = inputSeqL, 
            #     initial_state = initialState, 
            #     swap_memory = True)           

            # self.postModules = None
            # if (config.controlPostRNN or config.selfAttentionMod == "POST"): # may not work well with dlogits
            #     self.postModules, _ = self.RNNLayer(cLogits, None, config.encDim, bi = False, 
            #         name = "decPostRNN", cellType = config.controlPostRNNmod)
            #     if config.controlPostRNN:
            #         logits = self.postModules
            #     self.postModules = tf.unstack(self.postModules, axis = 1)

            # self.autoEncCtrlLoss = tf.constant(0.0)
            # if config.autoEncCtrl:
            #     autoEncCtrlCellType = ("GRU" if config.autoEncCtrlGRU else "RNN")
            #     autoEncCtrlinp = logits
            #     _, autoEncHid = self.RNNLayer(autoEncCtrlinp, None, config.encDim, 
            #       bi = True, name = "autoEncCtrl", cellType = autoEncCtrlCellType)
            #     self.autoEncCtrlLoss = (tf.nn.l2_loss(vecQuestions - autoEncHid)) / tf.to_float(self.batchSize)

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
    def outputOp(self, memory, vecQuestions, images, imageInDim):
        with tf.compat.v1.variable_scope("outputUnit"):
            features = memory
            dim = config.memDim

            if config.outQuestion:
                eVecQuestions = ops.linear(vecQuestions, config.ctrlDim, config.memDim, name = "outQuestion") 
                features, dim = ops.concat(features, eVecQuestions, config.memDim, mul = config.outQuestionMul)
            
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
    def classifier(self, features, inDim, aEmbeddings = None):
        with tf.compat.v1.variable_scope("classifier"):
            outDim = config.answerWordsNum
            dims = [inDim] + config.outClassifierDims + [outDim]
            if config.answerMod != "NON":
                dims[-1] = config.wrdEmbDim                


            logits = ops.FCLayer(features, dims, 
                batchNorm = self.batchNorm if config.outputBN else None, 
                dropout = self.dropouts["output"]) 
            
            if config.answerMod != "NON":
                logits = tf.compat.v1.nn.dropout(logits, self.dropouts["output"])
                interactions = ops.mul(aEmbeddings, logits, dims[-1], interMod = config.answerMod)
                logits = ops.inter2logits(interactions, dims[-1], sumMod = "SUM")
                logits += ops.getBias((outDim, ), "ans")

                # answersWeights = tf.transpose(aEmbeddings)

                # if config.answerMod == "BL":
                #     Wans = ops.getWeight((dims[-1], config.wrdEmbDim), "ans")
                #     logits = tf.matmul(logits, Wans)
                # elif config.answerMod == "DIAG":
                #     Wans = ops.getWeight((config.wrdEmbDim, ), "ans")
                #     logits = logits * Wans
                
                # logits = tf.matmul(logits, answersWeights) 

        return logits

    # def getTemp():
    #     with tf.compat.v1.variable_scope("temperature"):
    #         if config.tempParametric:
    #             self.temperatureVar = tf.compat.v1.get_variable("temperature", shape = (),
    #                 initializer = tf.constant_initializer(5), dtype = tf.float32)
    #             temperature = tf.sigmoid(self.temperatureVar)
    #         else:
    #             temperature = config.temperature
            
    #         if config.tempDynamic:
    #             temperature *= self.tempAnnealRate

    #     return temperature 

    # Computes mean cross entropy loss between logits and answers.
    def addAnswerLossOp(self, logits, answers):
        with tf.compat.v1.variable_scope("answerLoss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = answers, logits = logits)
            loss = tf.reduce_mean(losses)
            self.answerLossList.append(loss)

        return loss, losses

    # Computes predictions (by finding maximal logit value, corresponding to highest probability)
    # and mean accuracy between predictions and answers. 
    def addPredOp(self, logits, answers):
        with tf.compat.v1.variable_scope("pred"):
            preds = tf.compat.v1.to_int32(tf.argmax(logits, axis = -1)) # tf.nn.softmax(
            corrects = tf.equal(preds, answers) 
            correctNum = tf.reduce_sum(tf.compat.v1.to_int32(corrects))
            acc = tf.reduce_mean(tf.compat.v1.to_float(corrects))
            self.correctNumList.append(correctNum) 
            self.answerAccList.append(acc)

        return preds, corrects, correctNum

    # Creates optimizer (adam)
    def addOptimizerOp(self): 
        with tf.compat.v1.variable_scope("trainAddOptimizer"):
            self.globalStep = tf.Variable(0, dtype = tf.int32, trainable = False, name = "globalStep") # init to 0 every run?
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.lr)

        return optimizer

    '''
    Computes gradients for all variables or subset of them, based on provided loss, 
    using optimizer.
    '''
    def computeGradients(self, optimizer, loss, trainableVars = None): # tf.trainable_variables()
        with tf.compat.v1.variable_scope("computeGradients"):
            if config.trainSubset:
                trainableVars = []
                allVars = tf.compat.v1.trainable_variables()
                for var in allVars:
                    if any((s in var.name) for s in config.varSubset):
                        trainableVars.append(var)

            gradients_vars = optimizer.compute_gradients(loss, trainableVars) 
        return gradients_vars

    '''
    Apply gradients. Optionally clip them, and update exponential moving averages 
    for parameters.
    '''
    def addTrainingOp(self, optimizer, gradients_vars):
        with tf.compat.v1.variable_scope("train"):
            gradients, variables = zip(*gradients_vars)
            norm = tf.compat.v1.global_norm(gradients)

            # gradient clipping
            if config.clipGradients:            
                clippedGradients, _ = tf.clip_by_global_norm(gradients, config.gradMaxNorm, use_norm = norm)
                gradients_vars = zip(clippedGradients, variables)

            # updates ops (for batch norm) and train op
            updateOps = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updateOps):
                train = optimizer.apply_gradients(gradients_vars, global_step = self.globalStep)

            # exponential moving average
            if config.useEMA:
                ema = tf.train.ExponentialMovingAverage(decay = config.emaDecayRate)
                maintainAveragesOp = ema.apply(tf.compat.v1.trainable_variables())

                with tf.control_dependencies([train]):
                    trainAndUpdateOp = tf.group(maintainAveragesOp)
                
                train = trainAndUpdateOp

                self.emaDict = ema.variables_to_restore()

        return train, norm

    # TODO (add back support for multi-gpu..)
    def averageAcrossTowers(self, gpusNum):
        self.lossAll = self.lossList[0]

        self.answerLossAll = self.answerLossList[0]
        self.correctNumAll = self.correctNumList[0]
        self.answerAccAll = self.answerAccList[0]
        self.predsAll = self.predsList[0]
        self.gradientVarsAll = self.gradientVarsList[0]

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
                pred = self.answerDict.decodeId(predictions[i])
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
    def runBatch(self, sess, data, images, train, getAtt = False):     
        data = self.trimData(data)

        trainOp = self.trainOp if train else self.noOp
        gradNormOp = self.gradNorm if train else self.noOp

        predsOp = (self.predsAll, self.correctNumAll, self.answerAccAll)
        
        attOp = self.macCell.attentions
        
        time0 = time.time()
        feed = self.createFeedDict(data, images, train) 

        time1 = time.time()
        _, loss, predsInfo, gradNorm, attentionMaps = sess.run(
            [trainOp, self.lossAll, predsOp, gradNormOp, attOp], 
            feed_dict = feed)
        
        time2 = time.time()  

        predsList = self.buildPredsList(data, predsInfo[0], attentionMaps if getAtt else None)

        return {"loss": loss,
                "correctNum": predsInfo[1],
                "acc": predsInfo[2], 
                "preds": predsList,
                "gradNorm": gradNorm if train else -1,
                "readTime": time1 - time0,
                "trainTime": time2 - time1}

    def build(self):
        self.addPlaceholders()
        self.optimizer = self.addOptimizerOp()

        self.gradientVarsList = []
        self.lossList = []

        self.answerLossList = []
        self.correctNumList = []
        self.answerAccList = []
        self.predsList = []

        with tf.compat.v1.variable_scope("macModel"):
            for i in range(config.gpusNum):
                with tf.device("/gpu:{}".format(i)):
                    with tf.name_scope("tower{}".format(i)) as scope:
                        self.initTowerBatch(i, config.gpusNum, self.batchSizeAll)

                        self.loss = tf.constant(0.0)

                        # embed questions words (and optionally answer words)
                        questionWords, qEmbeddings, aEmbeddings = \
                            self.embeddingsOp(self.questionsIndices, self.embeddingsInit)

                        projWords = projQuestion = ((config.encDim != config.ctrlDim) or config.encProj)
                        questionCntxWords, vecQuestions = self.encoder(questionWords, 
                            self.questionLengths, projWords, projQuestion, config.ctrlDim)

                        # Image Input Unit (stem)
                        imageFeatures = self.stem(self.images, self.imageInDim, config.memDim)

                        # baseline model
                        if config.useBaseline:
                            output, dim = self.baseline(vecQuestions, config.ctrlDim, 
                                self.images, self.imageInDim, config.attDim)
                        # MAC model
                        else:      
                            # self.temperature = self.getTemp()
                            
                            finalControl, finalMemory = self.MACnetwork(imageFeatures, vecQuestions, 
                                questionWords, questionCntxWords, self.questionLengths)
                            
                            # Output Unit - step 1 (preparing classifier inputs)
                            output, dim = self.outputOp(finalMemory, vecQuestions, 
                                self.images, self.imageInDim)

                        # Output Unit - step 2 (classifier)
                        logits = self.classifier(output, dim, aEmbeddings)

                        # compute loss, predictions, accuracy
                        answerLoss, self.losses = self.addAnswerLossOp(logits, self.answersIndices)
                        self.preds, self.corrects, self.correctNum = self.addPredOp(logits, self.answersIndices)
                        self.loss += answerLoss
                        self.predsList.append(self.preds)

                        self.lossList.append(self.loss)

                        # compute gradients
                        gradient_vars = self.computeGradients(self.optimizer, self.loss, trainableVars = None)
                        self.gradientVarsList.append(gradient_vars)

                        # reuse variables in next towers
                        tf.compat.v1.get_variable_scope().reuse_variables()

        self.averageAcrossTowers(config.gpusNum)

        self.trainOp, self.gradNorm = self.addTrainingOp(self.optimizer, self.gradientVarsAll)
        self.noOp = tf.no_op()
