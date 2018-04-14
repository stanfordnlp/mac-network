from __future__ import division
import math
import tensorflow as tf

from mi_gru_cell import MiGRUCell
from mi_lstm_cell import MiLSTMCell
from config import config

eps = 1e-20
inf = 1e30

####################################### variables ########################################

'''
Initializes a weight matrix variable given a shape and a name. 
Uses random_normal initialization if 1d, otherwise uses xavier. 
'''
def getWeight(shape, name = ""):
    with tf.variable_scope("weights"):               
        initializer = tf.contrib.layers.xavier_initializer()
        # if len(shape) == 1: # good?
        #     initializer = tf.random_normal_initializer()        
        W = tf.get_variable("weight" + name, shape = shape, initializer = initializer)
    return W

'''
Initializes a weight matrix variable given a shape and a name. Uses xavier
'''
def getKernel(shape, name = ""):
    with tf.variable_scope("kernels"):               
        initializer = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable("kernel" + name, shape = shape, initializer = initializer)
    return W

'''
Initializes a bias variable given a shape and a name.
'''
def getBias(shape, name = ""):
    with tf.variable_scope("biases"):              
        initializer = tf.zeros_initializer()
        b = tf.get_variable("bias" + name, shape = shape, initializer = initializer)
    return b

######################################### basics #########################################

'''
Multiplies input inp of any depth by a 2d weight matrix.  
'''
# switch with conv 1?
def multiply(inp, W):
    inDim = tf.shape(W)[0]
    outDim = tf.shape(W)[1] 
    newDims = tf.concat([tf.shape(inp)[:-1], tf.fill((1,), outDim)], axis = 0)
    
    inp = tf.reshape(inp, (-1, inDim))
    output = tf.matmul(inp, W)
    output = tf.reshape(output, newDims)

    return output

'''
Concatenates x and y. Support broadcasting. 
Optionally concatenate multiplication of x * y
'''
def concat(x, y, dim, mul = False, extendY = False):
    if extendY:
        y = tf.expand_dims(y, axis = -2)
        # broadcasting to have the same shape
        y = tf.zeros_like(x) + y

    if mul:
        out = tf.concat([x, y, x * y], axis = -1)
        dim *= 3
    else:
        out = tf.concat([x, y], axis = -1)
        dim *= 2
    
    return out, dim

'''
Adds L2 regularization for weight and kernel variables.
'''
# add l2 in the tf way
def L2RegularizationOp(l2 = None):
    if l2 is None:
        l2 = config.l2
    l2Loss = 0
    names = ["weight", "kernel"]
    for var in tf.trainable_variables():
        if any((name in var.name.lower()) for name in names):
            l2Loss += tf.nn.l2_loss(var)
    return l2 * l2Loss

######################################### attention #########################################

'''
Transform vectors to scalar logits.

Args:
    interactions: input vectors
    [batchSize, N, dim]

    dim: dimension of input vectors

    sumMod: LIN for linear transformation to scalars.
            SUM to sum up vectors entries to get scalar logit.

    dropout: dropout value over inputs (for linear case)

Return matching scalar for each interaction.
[batchSize, N]
'''
sumMod = ["LIN", "SUM"]
def inter2logits(interactions, dim, sumMod = "LIN", dropout = 1.0, name = "", reuse = None):
    with tf.variable_scope("inter2logits" + name, reuse = reuse): 
        if sumMod == "SUM":
            logits = tf.reduce_sum(interactions, axis = -1)
        else: # "LIN"
            logits = linear(interactions, dim, 1, dropout = dropout, name = "logits")
    return logits

'''
Transforms vectors to probability distribution. 
Calls inter2logits and then softmax over these.

Args:
    interactions: input vectors
    [batchSize, N, dim]

    dim: dimension of input vectors

    sumMod: LIN for linear transformation to scalars.
            SUM to sum up vectors entries to get scalar logit.

    dropout: dropout value over inputs (for linear case)

Return attention distribution over interactions.
[batchSize, N]
'''
def inter2att(interactions, dim, dropout = 1.0, name = "", reuse = None):
    with tf.variable_scope("inter2att" + name, reuse = reuse): 
        logits = inter2logits(interactions, dim, dropout = dropout)
        attention = tf.nn.softmax(logits)    
    return attention

'''
Sums up features using attention distribution to get a weighted average over them. 
'''
def att2Smry(attention, features):
    return tf.reduce_sum(tf.expand_dims(attention, axis = -1) * features, axis = -2)

####################################### activations ########################################

'''
Performs a variant of ReLU based on config.relu
    PRM for PReLU
    ELU for ELU
    LKY for Leaky ReLU
    otherwise, standard ReLU
'''
def relu(inp):                  
    if config.relu == "PRM":
        with tf.variable_scope(None, default_name = "prelu"):
            alpha = tf.get_variable("alpha", shape = inp.get_shape()[-1], 
                initializer = tf.constant_initializer(0.25))
            pos = tf.nn.relu(inp)
            neg = - (alpha * tf.nn.relu(-inp))
            output = pos + neg
    elif config.relu == "ELU":
        output = tf.nn.elu(inp)
    # elif config.relu == "SELU":
    #     output = tf.nn.selu(inp) 
    elif config.relu == "LKY":
        # output = tf.nn.leaky_relu(inp, config.reluAlpha)
        output = tf.maximum(inp, config.reluAlpha * inp)
    elif config.relu == "STD": # STD
        output = tf.nn.relu(inp)
    
    return output

activations = {
    "NON":      tf.identity, # lambda inp: inp    
    "TANH":     tf.tanh,
    "SIGMOID":  tf.sigmoid,
    "RELU":     relu,
    "ELU":      tf.nn.elu
}    

# Sample from Gumbel(0, 1)
def sampleGumbel(shape): 
    U = tf.random_uniform(shape, minval = 0, maxval = 1)
    return -tf.log(-tf.log(U + eps) + eps)

# Draw a sample from the Gumbel-Softmax distribution
def gumbelSoftmaxSample(logits, temperature): 
    y = logits + sampleGumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def gumbelSoftmax(logits, temperature, train): # hard = False
    # Sample from the Gumbel-Softmax distribution and optionally discretize.
    # Args:
    #    logits: [batch_size, n_class] unnormalized log-probs
    #    temperature: non-negative scalar
    #    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    # Returns:
    #    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    #    If hard=True, then the returned sample will be one-hot, otherwise it will
    #    be a probabilitiy distribution that sums to 1 across classes

    y = gumbelSoftmaxSample(logits, temperature)

    # k = tf.shape(logits)[-1]
    # yHard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    yHard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims = True)), y.dtype)
    yNew = tf.stop_gradient(yHard - y) + y

    if config.gumbelSoftmaxBoth:
        return y
    if config.gumbelArgmaxBoth:
        return yNew
    ret = tf.cond(train, lambda: y, lambda: yNew)
    
    return ret 

def softmaxDiscrete(logits, temperature, train):
    if config.gumbelSoftmax:
        return gumbelSoftmax(logits, temperature = temperature, train = train)
    else:
        return tf.nn.softmax(logits)

def parametricDropout(name, train):
    var = tf.get_variable("varDp" + name, shape = (), initializer = tf.constant_initializer(2), 
        dtype = tf.float32)
    dropout = tf.cond(train, lambda: tf.sigmoid(var), lambda: 1.0)
    return dropout

###################################### sequence helpers ######################################

'''
Casts exponential mask over a sequence with sequence length.
Used to prepare logits before softmax.
'''
def expMask(seq, seqLength):
    maxLength = tf.shape(seq)[-1]
    mask = (1 - tf.cast(tf.sequence_mask(seqLength, maxLength), tf.float32)) * (-inf)
    masked = seq + mask
    return masked

'''
Computes seq2seq loss between logits and target sequences, with given lengths.
'''
def seq2SeqLoss(logits, targets, lengths):
    mask = tf.sequence_mask(lengths, maxlen = tf.shape(targets)[1])
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.to_float(mask))
    return loss

'''
Computes seq2seq loss between logits and target sequences, with given lengths.
    acc1: accuracy per symbol 
    acc2: accuracy per sequence
'''
def seq2seqAcc(preds, targets, lengths):
    mask = tf.sequence_mask(lengths, maxlen = tf.shape(targets)[1])
    corrects = tf.logical_and(tf.equal(preds, targets), mask)
    numCorrects = tf.reduce_sum(tf.to_int32(corrects), axis = 1)
    
    acc1 = tf.to_float(numCorrects) / (tf.to_float(lengths) + eps) # add small eps instead?
    acc1 = tf.reduce_mean(acc1)  
    
    acc2 = tf.to_float(tf.equal(numCorrects, lengths))
    acc2 = tf.reduce_mean(acc2)      

    return acc1, acc2

########################################### linear ###########################################

'''
linear transformation.

Args:
    inp: input to transform
    inDim: input dimension
    outDim: output dimension
    dropout: dropout over input
    batchNorm: if not None, applies batch normalization to inputs
    addBias: True to add bias
    bias: initial bias value
    act: if not None, activation to use after linear transformation
    actLayer: if True and act is not None, applies another linear transformation on top of previous
    actDropout: dropout to apply in the optional second linear transformation
    retVars: if True, return parameters (weight and bias) 

Returns linear transformation result.
'''
# batchNorm = {"decay": float, "train": Tensor}
# actLayer: if activation is not non, stack another linear layer
# maybe change naming scheme such that if name = "" than use it as default_name (-->unique?)
def linear(inp, inDim, outDim, dropout = 1.0, 
    batchNorm = None, addBias = True, bias = 0.0,
    act = "NON", actLayer = True, actDropout = 1.0, 
    retVars = False, name = "", reuse = None):
    
    with tf.variable_scope("linearLayer" + name, reuse = reuse):        
        W = getWeight((inDim, outDim) if outDim > 1 else (inDim, ))
        b = getBias((outDim, ) if outDim > 1 else ()) + bias
        
        if batchNorm is not None:
            inp = tf.contrib.layers.batch_norm(inp, decay = batchNorm["decay"], 
                center = True, scale = True, is_training = batchNorm["train"], updates_collections = None)
            # tf.layers.batch_normalization, axis -1 ?

        inp = tf.nn.dropout(inp, dropout)                
        
        if outDim > 1:
            output = multiply(inp, W)
        else:
            output = tf.reduce_sum(inp * W, axis = -1)
        
        if addBias:
            output += b

        output = activations[act](output)

        # good?
        if act != "NON" and actLayer:
            output = linear(output, outDim, outDim, dropout = actDropout, batchNorm = batchNorm,  
                addBias = addBias, act = "NON", actLayer = False, 
                name = name + "_2", reuse = reuse)

    if retVars:
        return (output, (W, b))

    return output

'''
Computes Multi-layer feed-forward network.

Args:
    features: input features
    dims: list with dimensions of network. 
          First dimension is of the inputs, final is of the outputs.
    batchNorm: if not None, applies batchNorm
    dropout: dropout value to apply for each layer
    act: activation to apply between layers.
    NON, TANH, SIGMOID, RELU, ELU
'''
# no activation after last layer
# batchNorm = {"decay": float, "train": Tensor}
def FCLayer(features, dims, batchNorm = None, dropout = 1.0, act = "RELU"):
    layersNum = len(dims) - 1
    
    for i in range(layersNum):
        features = linear(features, dims[i], dims[i+1], name = "fc_%d" % i, 
            batchNorm = batchNorm, dropout = dropout)
        # not the last layer
        if i < layersNum - 1: 
            features = activations[act](features)
    
    return features   

###################################### cnns ######################################

'''
Computes convolution.

Args:
    inp: input features
    inDim: input dimension
    outDim: output dimension
    batchNorm: if not None, applies batchNorm on inputs
    dropout: dropout value to apply on inputs
    addBias: True to add bias
    kernelSize: kernel size
    stride: stride size
    act: activation to apply on outputs
    NON, TANH, SIGMOID, RELU, ELU
'''
# batchNorm = {"decay": float, "train": Tensor, "center": bool, "scale": bool}
# collections.namedtuple("batchNorm", ("decay", "train"))
def cnn(inp, inDim, outDim, batchNorm = None, dropout = 1.0, addBias = True, 
    kernelSize = None, stride = 1, act = "NON", name = "", reuse = None):
    
    with tf.variable_scope("cnnLayer" + name, reuse = reuse):
        
        if kernelSize is None:
            kernelSize = config.stemKernelSize            
        kernelH = kernelW = kernelSize
        
        kernel = getKernel((kernelH, kernelW, inDim, outDim))
        b = getBias((outDim, ))
        
        if batchNorm is not None:
            inp = tf.contrib.layers.batch_norm(inp, decay = batchNorm["decay"], center = batchNorm["center"], 
                scale = batchNorm["scale"], is_training = batchNorm["train"], updates_collections = None)   

        inp = tf.nn.dropout(inp, dropout)                
        
        output = tf.nn.conv2d(inp, filter = kernel, strides = [1, stride, stride, 1], padding = "SAME")
        
        if addBias:
            output += b

        output = activations[act](output)

    return output

'''
Computes Multi-layer convolutional network.

Args:
    features: input features
    dims: list with dimensions of network. 
          First dimension is of the inputs. Final is of the outputs.
    batchNorm: if not None, applies batchNorm
    dropout: dropout value to apply for each layer
    kernelSizes: list of kernel sizes for each layer. Default to config.stemKernelSize
    strides: list of strides for each layer. Default to 1.
    act: activation to apply between layers.
    NON, TANH, SIGMOID, RELU, ELU
'''
# batchNorm = {"decay": float, "train": Tensor, "center": bool, "scale": bool}
# activation after last layer
def CNNLayer(features, dims, batchNorm = None, dropout = 1.0, 
    kernelSizes = None, strides = None, act = "RELU"):
    
    layersNum = len(dims) - 1
    
    if kernelSizes is None:
        kernelSizes = [config.stemKernelSize for i in range(layersNum)]
    
    if strides is None:
        strides = [1 for i in range(layersNum)]
    
    for i in range(layersNum):
        features = cnn(features, dims[i], dims[i+1], name = "cnn_%d" % i, batchNorm = batchNorm, 
            dropout = dropout, kernelSize = kernelSizes[i], stride = strides[i], act = act)

    return features

######################################## location ########################################

'''
Computes linear positional encoding for h x w grid. 
If outDim positive, casts positions to that dimension.
'''
# ignores dim
# h,w can be tensor scalars
def locationL(h, w, dim, outDim = -1, addBias = True):
    dim = 2
    grid = tf.stack(tf.meshgrid(tf.linspace(-config.locationBias, config.locationBias, w), 
                                tf.linspace(-config.locationBias, config.locationBias, h)), axis = -1)

    if outDim > 0:
        grid = linear(grid, dim, outDim, addBias = addBias, name = "locationL")
        dim = outDim

    return grid, dim

'''
Computes sin/cos positional encoding for h x w x (4*dim). 
If outDim positive, casts positions to that dimension.
Based on positional encoding presented in "Attention is all you need"
'''
# dim % 4 = 0
# h,w can be tensor scalars
def locationPE(h, w, dim, outDim = -1, addBias = True):    
    x = tf.expand_dims(tf.to_float(tf.linspace(-config.locationBias, config.locationBias, w)), axis = -1)
    y = tf.expand_dims(tf.to_float(tf.linspace(-config.locationBias, config.locationBias, h)), axis = -1)
    i = tf.expand_dims(tf.to_float(tf.range(dim)), axis = 0)

    peSinX = tf.sin(x / (tf.pow(10000.0, i / dim)))
    peCosX = tf.cos(x / (tf.pow(10000.0, i / dim)))
    peSinY = tf.sin(y / (tf.pow(10000.0, i / dim)))
    peCosY = tf.cos(y / (tf.pow(10000.0, i / dim)))

    peSinX = tf.tile(tf.expand_dims(peSinX, axis = 0), [h, 1, 1])
    peCosX = tf.tile(tf.expand_dims(peCosX, axis = 0), [h, 1, 1])
    peSinY = tf.tile(tf.expand_dims(peSinY, axis = 1), [1, w, 1])
    peCosY = tf.tile(tf.expand_dims(peCosY, axis = 1), [1, w, 1]) 

    grid = tf.concat([peSinX, peCosX, peSinY, peCosY], axis = -1)
    dim *= 4
    
    if outDim > 0:
        grid = linear(grid, dim, outDim, addBias = addBias, name = "locationPE")
        dim = outDim

    return grid, dim

locations = {
    "L": locationL,
    "PE": locationPE
}

'''
Adds positional encoding to features. May ease spatial reasoning.
(although not used in the default model). 

Args:
    features: features to add position encoding to.
    [batchSize, h, w, c]

    inDim: number of features' channels
    lDim: dimension for positional encodings
    outDim: if positive, cast enhanced features (with positions) to that dimension
    h: features' height
    w: features' width
    locType: L for linear encoding, PE for cos/sin based positional encoding
    mod: way to add positional encoding: concatenation (CNCT), addition (ADD), 
            multiplication (MUL), linear transformation (LIN).
'''
mods = ["CNCT", "ADD", "LIN", "MUL"]
# if outDim = -1, then will be set based on inDim, lDim
def addLocation(features, inDim, lDim, outDim = -1, h = None, w = None, 
    locType = "L", mod = "CNCT", name = "", reuse = None): # h,w not needed
    
    with tf.variable_scope("addLocation" + name, reuse = reuse):
        batchSize = tf.shape(features)[0]
        if h is None:
            h = tf.shape(features)[1]
        if w is None:
            w = tf.shape(features)[2]
        dim = inDim

        if mod == "LIN":
            if outDim < 0:
                outDim = dim

            grid, _ = locations[locType](h, w, lDim, outDim = outDim, addBias = False)
            features = linear(features, dim, outDim, name = "LIN")
            features += grid  
            return features, outDim

        if mod == "CNCT":
            grid, lDim = locations[locType](h, w, lDim)
            # grid = tf.zeros_like(features) + grid
            grid = tf.tile(tf.expand_dims(grid, axis = 0), [batchSize, 1, 1, 1])
            features = tf.concat([features, grid], axis = -1)
            dim += lDim

        elif mod == "ADD":
            grid, _ = locations[locType](h, w, lDim, outDim = dim)
            features += grid    
        
        elif mod == "MUL": # MUL
            grid, _ = locations[locType](h, w, lDim, outDim = dim)

            if outDim < 0:
                outDim = dim

            grid = tf.tile(tf.expand_dims(grid, axis = 0), [batchSize, 1, 1, 1])
            features = tf.concat([features, grid, features * grid], axis = -1)
            dim *= 3                

        if outDim > 0:
            features = linear(features, dim, outDim)
            dim = outDim 

    return features, dim

# config.locationAwareEnd
# H, W, _ = config.imageDims
# projDim = config.stemProjDim
# k = config.stemProjPooling
# projDim on inDim or on out
# inDim = tf.shape(features)[3]

'''
Linearize 2d image to linear vector.

Args:
    features: batch of 2d images. 
    [batchSize, h, w, inDim]

    h: image height

    w: image width

    inDim: number of channels

    projDim: if not None, project image to that dimension before linearization

    outDim: if not None, project image to that dimension after linearization

    loc: if not None, add positional encoding:
        locType: L for linear encoding, PE for cos/sin based positional encoding
        mod: way to add positional encoding: concatenation (CNCT), addition (ADD), 
            multiplication (MUL), linear transformation (LIN).
        pooling: number to pool image with before linearization.

Returns linearized image:
[batchSize, outDim] (or [batchSize, (h / pooling) * (w /pooling) * projDim] if outDim not supported) 
'''
# loc = {"locType": str, "mod": str}
def linearizeFeatures(features, h, w, inDim, projDim = None, outDim = None, 
    loc = None, pooling = None):
    
    if pooling is None:
        pooling = config.imageLinPool
    
    if loc is not None:
        features = addLocation(features, inDim, lDim = inDim, outDim = inDim, 
            locType = loc["locType"], mod = loc["mod"])

    if projDim is not None:
        features = linear(features, dim, projDim)
        features = relu(features)
        dim = projDim

    if pooling > 1:
        poolingDims = [1, pooling, pooling, 1]
        features = tf.nn.max_pool(features, ksize = poolingDims, strides = poolingDims, 
            padding = "SAME")
        h /= pooling
        w /= pooling
  
    dim = h * w * dim  
    features = tf.reshape(features, (-1, dim))
    
    if outDim is not None:
        features = linear(features, dim, outDim)
        dim = outDim

    return features, dim

################################### multiplication ###################################
# specific dim / proj for x / y
'''
"Enhanced" hadamard product between x and y:
1. Supports optional projection of x, and y prior to multiplication.
2. Computes simple multiplication, or a parametrized one, using diagonal of complete matrix (bi-linear) 
3. Optionally concatenate x or y or their projection to the multiplication result.

Support broadcasting

Args:
    x: left-hand side argument
    [batchSize, dim]

    y: right-hand side argument
    [batchSize, dim]

    dim: input dimension of x and y
    
    dropout: dropout value to apply on x and y

    proj: if not None, project x and y:
        dim: projection dimension
        shared: use same projection for x and y
        dropout: dropout to apply to x and y if projected

    interMod: multiplication type:
        "MUL": x * y
        "DIAG": x * W * y for a learned diagonal parameter W
        "BL": x' W y for a learned matrix W

    concat: if not None, concatenate x or y or their projection. 
    
    mulBias: optional bias to stabilize multiplication (x * bias) (y * bias)

Returns the multiplication result
[batchSize, outDim] when outDim depends on the use of proj and cocnat arguments.
'''
# proj = {"dim": int, "shared": bool, "dropout": float} # "act": str, "actDropout": float
## interMod = ["direct", "scalarW", "bilinear"] # "additive"
# interMod = ["MUL", "DIAG", "BL", "ADD"]
# concat = {"x": bool, "y": bool, "proj": bool}
def mul(x, y, dim, dropout = 1.0, proj = None, interMod = "MUL", concat = None, mulBias = None,
    extendY = True, name = "", reuse = None):
    
    with tf.variable_scope("mul" + name, reuse = reuse):                
        origVals = {"x": x, "y": y, "dim": dim}

        x = tf.nn.dropout(x, dropout)
        y = tf.nn.dropout(y, dropout)
        # projection
        if proj is not None:
            x = tf.nn.dropout(x, proj.get("dropout", 1.0))
            y = tf.nn.dropout(y, proj.get("dropout", 1.0))

            if proj["shared"]:
                xName, xReuse = "proj", None
                yName, yReuse = "proj", True
            else:
                xName, xReuse = "projX", None
                yName, yReuse = "projY", None

            x = linear(x, dim, proj["dim"], name = xName, reuse = xReuse)
            y = linear(y, dim, proj["dim"], name = yName, reuse = yReuse)
            dim = proj["dim"]
            projVals = {"x": x, "y": y, "dim": dim}
            proj["x"], proj["y"] = x, y

        if extendY:
            y = tf.expand_dims(y, axis = -2)
            # broadcasting to have the same shape
            y = tf.zeros_like(x) + y

        # multiplication
        if interMod == "MUL":
            if mulBias is None:
                mulBias = config.mulBias             
            output = (x + mulBias) * (y + mulBias)
        elif interMod == "DIAG":
            W = getWeight((dim, )) # change initialization?
            b = getBias((dim, ))
            activations = x * W * y + b
        elif interMod == "BL":
            W = getWeight((dim, dim))
            b = getBias((dim, ))            
            output = multiply(x, W) * y + b
        else: # "ADD"
            output = tf.tanh(x + y)
        # concatenation
        if concat is not None:
            concatVals = projVals if concat.get("proj", False) else origVals
            if concat.get("x", False):
                output = tf.concat([output, concatVals["x"]], axis = -1)
                dim += concatVals["dim"]

            if concat.get("y", False):
                output = ops.concat(output, concatVals["y"], extendY = extendY)
                dim += concatVals["dim"]

    return output, dim

######################################## rnns ########################################

'''
Creates an RNN cell.

Args:
    hdim: the hidden dimension of the RNN cell.
    
    reuse: whether the cell should reuse parameters or create new ones.
    
    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM, ProjLSTM

    act: the cell activation
    NON, TANH, SIGMOID, RELU, ELU

    projDim: if ProjLSTM, the dimension for the states projection

Returns the cell.
'''
# tf.nn.rnn_cell.MultiRNNCell([cell(hDim, reuse = reuse) for _ in config.encNumLayers])
# note that config.enc params not general 
def createCell(hDim, reuse, cellType = None, act = None, projDim = None):
    if cellType is None:
        cellType = config.encType

    activation = activations.get(act, None) 

    if cellType == "ProjLSTM":
        cell = tf.nn.rnn_cell.LSTMCell
        if projDim is None:
            projDim = config.cellDim
        cell = cell(hDim, num_proj = projDim, reuse = reuse, activation = activation)
        return cell        

    cells = {
        "RNN": tf.nn.rnn_cell.BasicRNNCell,
        "GRU": tf.nn.rnn_cell.GRUCell,
        "LSTM": tf.nn.rnn_cell.BasicLSTMCell,
        "MiGRU": MiGRUCell,
        "MiLSTM": MiLSTMCell
    }

    cell = cells[cellType](hDim, reuse = reuse, activation = activation)

    return cell

'''
Runs an forward RNN layer.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM, ProjLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize). 

Returns the outputs sequence and final RNN state.  
'''
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
def fwRNNLayer(inSeq, seqL, hDim, cellType = None, dropout = 1.0, varDp = None, 
    name = "", reuse = None): # proj = None
    
    with tf.variable_scope("rnnLayer" + name, reuse = reuse):
        batchSize = tf.shape(inSeq)[0]

        cell = createCell(hDim, reuse, cellType) # passing reuse isn't mandatory

        if varDp is not None:
            cell = tf.contrib.rnn.DropoutWrapper(cell, 
                state_keep_prob = varDp["stateDp"],
                input_keep_prob = varDp["inputDp"],
                variational_recurrent = True, input_size = varDp["inputSize"], dtype = tf.float32)
        else:           
            inSeq = tf.nn.dropout(inSeq, dropout)
        
        initialState = cell.zero_state(batchSize, tf.float32)

        outSeq, lastState = tf.nn.dynamic_rnn(cell, inSeq, 
            sequence_length = seqL, 
            initial_state = initialState,
            swap_memory = True)
            
        if isinstance(lastState, tf.nn.rnn_cell.LSTMStateTuple):
            lastState = lastState.h

        # if proj is not None:
        #     if proj["output"]:
        #         outSeq = linear(outSeq, cell.output_size, proj["dim"], act = proj["act"],  
        #             dropout = proj["dropout"], name = "projOutput")

        #     if proj["state"]:
        #         lastState = linear(lastState, cell.state_size, proj["dim"], act = proj["act"],  
        #             dropout = proj["dropout"], name = "projState")

    return outSeq, lastState

'''
Runs an bidirectional RNN layer.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize).   

Returns the outputs sequence and final RNN state.     
'''
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
def biRNNLayer(inSeq, seqL, hDim, cellType = None, dropout = 1.0, varDp = None, 
    name = "", reuse = None): # proj = None, 

    with tf.variable_scope("birnnLayer" + name, reuse = reuse):
        batchSize = tf.shape(inSeq)[0]

        with tf.variable_scope("fw"):
            cellFw = createCell(hDim, reuse, cellType)
        with tf.variable_scope("bw"):
            cellBw = createCell(hDim, reuse, cellType)
        
        if varDp is not None:
            cellFw = tf.contrib.rnn.DropoutWrapper(cellFw, 
                state_keep_prob = varDp["stateDp"],
                input_keep_prob = varDp["inputDp"],
                variational_recurrent = True, input_size = varDp["inputSize"], dtype = tf.float32)
            
            cellBw = tf.contrib.rnn.DropoutWrapper(cellBw, 
                state_keep_prob = varDp["stateDp"],
                input_keep_prob = varDp["inputDp"],
                variational_recurrent = True, input_size = varDp["inputSize"], dtype = tf.float32)            
        else:
            inSeq = tf.nn.dropout(inSeq, dropout)

        initialStateFw = cellFw.zero_state(batchSize, tf.float32)
        initialStateBw = cellBw.zero_state(batchSize, tf.float32)

        (outSeqFw, outSeqBw), (lastStateFw, lastStateBw) = tf.nn.bidirectional_dynamic_rnn(
            cellFw, cellBw, inSeq, 
            sequence_length = seqL, 
            initial_state_fw = initialStateFw, 
            initial_state_bw = initialStateBw,
            swap_memory = True)

        if isinstance(lastStateFw, tf.nn.rnn_cell.LSTMStateTuple):
            lastStateFw = lastStateFw.h # take c? 
            lastStateBw = lastStateBw.h  

        outSeq = tf.concat([outSeqFw, outSeqBw], axis = -1)
        lastState = tf.concat([lastStateFw, lastStateBw], axis = -1)

        # if proj is not None:
        #     if proj["output"]:
        #         outSeq = linear(outSeq, cellFw.output_size + cellFw.output_size, 
        #             proj["dim"], act = proj["act"], dropout = proj["dropout"], 
        #             name = "projOutput")

        #     if proj["state"]:
        #         lastState = linear(lastState, cellFw.state_size + cellFw.state_size, 
        #             proj["dim"], act = proj["act"], dropout = proj["dropout"], 
        #             name = "projState")

    return outSeq, lastState

# int(hDim / 2) for biRNN?
'''
Runs an RNN layer by calling biRNN or fwRNN.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    bi: true to run bidirectional rnn.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize).   

Returns the outputs sequence and final RNN state.     
'''
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
def RNNLayer(inSeq, seqL, hDim, bi = None, cellType = None, dropout = 1.0, varDp = None, 
    name = "", reuse = None): # proj = None
    
    with tf.variable_scope("rnnLayer" + name, reuse = reuse):
        if bi is None:
            bi = config.encBi
        
        rnn = biRNNLayer if bi else fwRNNLayer
        
        if bi:
            hDim = int(hDim / 2)

    return rnn(inSeq, seqL, hDim, cellType = cellType, dropout = dropout, varDp = varDp) # , proj = proj

# tf counterpart?
# hDim = config.moduleDim
def multigridRNNLayer(featrues, h, w, dim, name = "", reuse = None):
    with tf.variable_scope("multigridRNNLayer" + name, reuse = reuse):
        featrues = linear(featrues, dim, dim / 2, name = "i")

        output0 = gridRNNLayer(featrues, h, w, dim, right = True, down = True, name = "rd")
        output1 = gridRNNLayer(featrues, h, w, dim, right = True, down = False, name = "r")
        output2 = gridRNNLayer(featrues, h, w, dim, right = False, down = True, name = "d")
        output3 = gridRNNLayer(featrues, h, w, dim, right = False, down = False, name = "NON")

        output = tf.concat([output0, output1, output2, output3], axis = -1)
        output = linear(output, 2 * dim, dim, name = "o")

    return outputs

# h,w should be constants
def gridRNNLayer(features, h, w, dim, right, down, name = "", reuse = None):
    with tf.variable_scope("gridRNNLayer" + name):
        batchSize = tf.shape(features)[0]

        cell = createCell(dim, reuse = reuse, cellType = config.stemGridRnnMod, 
            act = config.stemGridAct)
        
        initialState = cell.zero_state(batchSize, tf.float32)
        
        inputs = [tf.unstack(row, w, axis = 1) for row in tf.unstack(features, h, axis = 1)]
        states = [[None for _ in range(w)] for _ in range(h)]

        iAxis = range(h) if down else (range(h)[::-1])
        jAxis = range(w) if right else (range(w)[::-1])

        iPrev = -1 if down else 1
        jPrev = -1 if right else 1

        prevState = lambda i,j: states[i][j] if (i >= 0 and i < h and j >= 0 and j < w) else initialState
        
        for i in iAxis:
            for j in jAxis:
                prevs = tf.concat((prevState(i + iPrev, j), prevState(i, j + jPrev)), axis = -1)
                curr = inputs[i][j]
                _, states[i][j] = cell(prevs, curr)

        outputs = [tf.stack(row, axis = 1) for row in states]
        outputs = tf.stack(outputs, axis = 1)

    return outputs

# tf seq2seq?
# def projRNNLayer(inSeq, seqL, hDim, labels, labelsNum, labelsDim, labelsEmb, name = "", reuse = None):
#     with tf.variable_scope("projRNNLayer" + name):
#         batchSize = tf.shape(features)[0]

#         cell = createCell(hDim, reuse = reuse)

#         projCell = ProjWrapper(cell, labelsNum, labelsDim, labelsEmb, # config.wrdEmbDim
#             feedPrev = True, dropout = 1.0, config,
#             temperature = 1.0, sample = False, reuse)
        
#         initialState = projCell.zero_state(batchSize, tf.float32)
        
#         if config.soft:
#             inSeq = inSeq

#             # outputs, _ = tf.nn.static_rnn(projCell, inputs, 
#             #     sequence_length = seqL, 
#             #     initial_state = initialState)

#             inSeq = tf.unstack(inSeq, axis = 1)                        
#             state = initialState
#             logitsList = []
#             chosenList = []

#             for inp in inSeq:
#                 (logits, chosen), state = projCell(inp, state)
#                 logitsList.append(logits)
#                 chosenList.append(chosen)
#                 projCell.reuse = True

#             logitsOut = tf.stack(logitsList, axis = 1)
#             chosenOut = tf.stack(chosenList, axis = 1)
#             outputs = (logitsOut, chosenOut)
#         else:
#             labels = tf.to_float(labels)
#             labels = tf.concat([tf.zeros((batchSize, 1)), labels], axis = 1)[:, :-1] # ,newaxis
#             inSeq = tf.concat([inSeq, tf.expand_dims(labels, axis = -1)], axis = -1)

#             outputs, _ = tf.nn.dynamic_rnn(projCell, inSeq, 
#                 sequence_length = seqL, 
#                 initial_state = initialState,
#                 swap_memory = True)

#     return outputs #, labelsEmb

############################### variational dropout ###############################

'''
Generates a variational dropout mask for a given shape and a dropout 
probability value.
'''
def generateVarDpMask(shape, keepProb):
    randomTensor = tf.to_float(keepProb)
    randomTensor += tf.random_uniform(shape, minval = 0, maxval = 1)
    binaryTensor = tf.floor(randomTensor)
    mask = tf.to_float(binaryTensor)
    return mask

'''
Applies the a variational dropout over an input, given dropout mask 
and a dropout probability value. 
'''
def applyVarDpMask(inp, mask, keepProb):
    ret = (tf.div(inp, tf.to_float(keepProb))) * mask
    return ret   
