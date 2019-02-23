# GQA: a new dataset for real-world visual reasoning

<p align="center">
  <b>Drew A. Hudson & Christopher D. Manning</b></span>
</p>

This is an extension of the [MAC network](https://arxiv.org/pdf/1803.03067.pdf) to work on the <b>[the GQA dataset](https://www.visualreasoning.net)</b>. GQA is a new dataset for real-world visual reasoning, offrering 20M diverse multi-step questions, all come along with short programs that represent their semantics, and visual pointers from words to the corresponding image regions. Here we extend the MAC network to work over VQA and GQA, and provide multiple baselines as well. 

MAC is a fully differentiable model that learns to perform multi-step reasoning. See our [website](https://cs.stanford.edu/people/dorarad/mac/) and [blogpost](https://cs.stanford.edu/people/dorarad/mac/blog.html) for more information about the model, and visit the [GQA website](https://www.visualreasoning.net) for all information about the new dataset, including examples, visualizations, paper and slides.

<div align="center">
  <img src="https://cs.stanford.edu/people/dorarad/mac/imgs/cell.png" style="float:left" width="420px">
  <img src="https://cs.stanford.edu/people/dorarad/visual2.png" style="float:right" width="390px">
</div>

## Requirements
- Tensorflow (originally has been developed with 1.3 but should work for later versions as well).
- We have performed experiments on Maxwell Titan X GPU. We assume 12GB of GPU memory.
- See [`requirements.txt`](requirements.txt) for the required python packages and run `pip install -r requirements.txt` to install them.

Let's begin from cloning this reponsitory branch:
```
git clone -b gqa https://github.com/stanfordnlp/mac-network.git
```

## Pre-processing
Before training the model, we first have to download the GQA dataset and extract features for the images:

### Dataset
To download and unpack the data, run the following commands:
```bash
mkdir data
cd data
wget https://nlp.stanford.edu/data/gqa/data.zip
unzip data.zip
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ../
```
We also download GloVe word embeddings which we will use in our model. The `data` directory will hold all the data files we use during training.

Note: `data.zip` matches the official dataset at [`visualreasoning.net`](https://www.visualreasoning.net/download.html), but, in order to save space, contains about each question only the information needed to train MAC (e.g. doesn't contain the functional programs).

### Feature extraction
Both spatial ResNet-101 features as well as object-based faster-RCNN features are available for the GQA train, val, and test images. Download, extract and merge the features through the following commands (22GB).

```bash
cd data
wget http://nlp.stanford.edu/data/gqa/spatialFeatures.zip
wget http://nlp.stanford.edu/data/gqa/objectFeatures.zip
unzip objectFeatures.zip
cd ../
python merge.py --name spatial 
python merge.py --name objects 
```

## Training 
To train the model, run the following command:
```bash
python main.py --expName "gqaExperiment" --train --testedNum 10000 --epochs 25 --netLength 4 @configs/gqa/gqa.txt
```

First, the program preprocesses the GQA questions. It tokenizes them and maps them to integers to prepare them for the network. It then stores a JSON with that information about them as well as word-to-integer dictionaries in the `data` directory.

Then, the program trains the model. Weights are saved by default to `./weights/{expName}` and statistics about the training are collected in `./results/{expName}`, where `expName` is the name we choose to give to the current experiment. 

### Notes
- The number of examples used for training and evaluation can be set by `--trainedNum` and `--testedNum` respectively.
- You can use the `-r` flag to restore and continue training a previously pre-trained model. 
- We recommend you to try out varying the number of MAC cells used in the network through the `--netLength` option to explore different lengths of reasoning processes.
- Good lengths for GQA are in the range of 2-6. 

See [`config.py`](config.py) for further available options (Note that some of them are still in an experimental stage).

## Baselines 
Other language and vision based baselines are available. Run them by the following commands:
```bash
python main.py --expName "gqaLSTM" --train --testedNum 10000 --epochs 25 @configs/gqa/gqaLSTM.txt
python main.py --expName "gqaCNN" --train --testedNum 10000 --epochs 25 @configs/gqa/gqaCNN.txt
python main.py --expName "gqaLSTM-CNN" --train --testedNum 10000 --epochs 25 @configs/gqa/gqaLSTMCNN.txt
```

## Evalutation
To evaluate the trained model, and get predictions and attention maps, run the following: 
```bash
python main.py --expName "gqaExperiment" --finalTest --testedNum 10000 --netLength 4 -r --getPreds --getAtt @configs/gqa/gqa.txt
```
The command will restore the model we have trained, and evaluate it on the validation set. JSON files with predictions and the attention distributions resulted by running the model are saved by default to `./preds/{expName}`.

- In case you are interested in getting attention maps (`--getAtt`), and to avoid having large prediction files, we advise you to limit the number of examples evaluated to 5,000-20,000.

## Bibtex
```
@article{hudson2018compositional,
  title={GQA: a New Dataset for Compositional Question Answering over Real-World Images},
  author={Hudson, Drew A and Manning, Christopher D},
  year={2018}
}
```

Thank you for your interest in our model and the dataset! Please contact me at dorarad@stanford.edu for any questions, comments, or suggestions! :-)
