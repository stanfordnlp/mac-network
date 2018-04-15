# Compostional Attention Networks for Machine Reasoning
<p align="center">
  <b>Drew A. Hudson & Christopher D. Manning</b></span>
</p>

This is the implementation of [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf) (ICLR 2018). We propose a fully differentiable model that learns to perform multi-step reasoning and explore it in the context of the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/).
See our [website](https://cs.stanford.edu/people/dorarad/mac/) and [blogpost](https://cs.stanford.edu/people/dorarad/mac/blog.html) for more information about the model!

In particular, the implementation includes the MAC cell at [`mac_cell.py`](mac_cell.py). The code supports the standard cell as presented in the paper as well as additional extensions and variants. Run `python main.py -h` or see [`config.py`](config.py) for the complete list of options.

<div align="center">
  <img src="https://cs.stanford.edu/people/dorarad/mac/imgs/cell.png" style="float:left" width="390px">
  <img src="https://cs.stanford.edu/people/dorarad/mac/imgs/visual.png" style="float:right" width="480px">
</div>

## Requirements
- Tensorflow (originally has been developed with 1.3 but should work for later versions as well).
- We have performed experiments on Maxwell Titan X GPU. We assume 12GB of GPU memory.
- See [`requirements.txt`](requirements.txt) for the required python packages and run `pip install -r requirements.txt` to install them.

## Pre-processing
Before training the model, we first have to download the CLEVR dataset and extract features for the images:

### Dataset
To download and unpack the data, run the following commands:
```bash
wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
mv CLEVR_v1.0 CLEVR_v1
mkdir CLEVR_v1/data
mv CLEVR_v1/questions/* CLEVR_v1/data/
```
The final command moves the dataset questions into the `data` directory, where we will put all the data files we use during training.

### Feature extraction
Extract ResNet-101 features for the CLEVR train, val, and test images with the following commands:

```bash
python extract_features.py --input_image_dir CLEVR_v1/images/train --output_h5_file CLEVR_v1/data/train.h5
python extract_features.py --input_image_dir CLEVR_v1/images/val --output_h5_file CLEVR_v1/data/val.h5
python extract_features.py --input_image_dir CLEVR_v1/images/test --output_h5_file CLEVR_v1/data/test.h5
```

## Training 
To train the model, run the following command:
```bash
python main.py --expName "clevrExperiment" --train --testedNum 10000 --epochs 25 --netLength 16 @configs/args.txt
```

First, the program preprocesses the CLEVR questions. It tokenizes them and maps them to integers to prepare them for the network. It then stores a JSON with that information about them as well as word-to-integer dictionaries in the `./CLEVR_v1/data` directory.

Then, the program trains the model. Weights are saved by default to `./weights/{expName}` and statistics about the training are collected in `./results/{expName}`, where `expName` is the name we choose to give to the current experiment. 

### Notes
- The number of examples used for training and evaluation can be set by `--trainedNum` and `--testedNum` respectively.
- You can use the `-r` flag to restore and continue training a previously pre-trained model. 
- We recommend you to try out varying the number of MAC cells used in the network through the `--netLength` option to explore different lengths of reasoning processes!
- Good lengths for CLEVR are in the range of 4-16 (using more cells tends to converge faster and achieve a bit higher accuracy, while lower number of cells usually result in more easily interpretable attention maps). 

### Model variants
We have explored several variants of our model. We provide a few examples in `configs/args1-4.txt`. For instance, you can run the first by: 
```bash
python main.py --expName "experiment1" --train --testedNum 10000 --epochs 25 --netLength 8 @configs/args1.txt
```
- [`args1`](config/args1.txt) is the standard recurrent-control-memory cell. Leads to the most interpretable results compared to others.
- [`args2`](config/args2.txt) uses a variant of the control unit that tends to converge fast and yield high accuracy.
- [`args3`](config/args3.txt) incorporates self-attention into the memory unit.
- [`args4`](config/args4.txt) adds memory control-based gating.

See [`config.py`](config.py) for further available options (Note that some of them are still in an experimental stage).

## Evalutation
To evaluate the trained model, and get predictions and attention maps, run the following: 
```bash
python main.py --expName "clevrExperiment" --finalTest --testedNum 10000 --netLength 16 -r --getPreds --getAtt @configs/args.txt
```
The command will restore the model we have trained, and evaluate it on the validation set. JSON files with predictions and the attention distributions resulted by running the model are saved by default to `./preds/{expName}`.

- In case you are interested in getting attention maps (`--getAtt`), and to avoid having large prediction files, we advise you to limit the number of examples evaluated to 5,000-20,000.

## Visualization
After we evaluate the model with the command above, we can visualize the attention maps generated by running:
```bash
python visualization.py --expName "clevrExperiment" --tier val 
```
(Tier can be set to `train` or `test` as well). The script supports filtering of the visualized questions by various ways. See [`visualization.py`](visualization.py) for further details.

Optionally, to make the image attention maps a little bit nicer, you can do the following (using [imagemagick](https://www.imagemagick.org)):
```
for x in preds/clevrExperiment/*Img*.png; do magick convert $x -brightness-contrast 20x35 $x; done;
```

## Bibtex
```
@inproceedings{hudson2018compositional,
  title={Compositional Attention Networks for Machine Reasoning},
  author={Hudson, Drew A and Manning, Christopher D},
  journal={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```

Thank you for your interest in our model! Please contact me at dorarad@cs.stanford.edu for any questions, comments, or suggestions! :-)
