# Glove-Torch

**[WIP]**An implementation of GloVe model in pytorch.

## Background

I planed to use the loss function of GloVe in a pytorch project so that I created this one to test the performance of the GloVe loss implemented in pytorch.

It has not reached the performance of [official GloVe model implementation by C](https://nlp.stanford.edu/projects/glove/) yet.

So it is still **WIP**.

If you found a miss in the codes, please raise an issue, thank you.

## Usage

### glove_torch_multithread_custom_loss.py 

It reads the vocab file and concurrent matrix file generate by [official GloVe model implementation by C](https://nlp.stanford.edu/projects/glove/).

`
python3 glove_torch_multithread_custom_loss.py --help
`
 gives you helps.
 
 ### https://github.com/yuanzhiKe/Glove-Torch/blob/master/output_vector.py
 
 Generate word vectors by adding wi.weight and wj.weight.
 
 `
python3 output_vector.py --help
`
 gives you helps.
 
 
## Features To-do

- [x] GloVe Loss
- [x] Multi-threading
- [ ] Reproduce performance of official implementation by C
