# Pytorch implementation of object discovery with slot attention

This repository is a pytorch implementation of the paper https://arxiv.org/abs/2006.15055

The official code for this paper can be found https://github.com/google-research/google-research/tree/bb24df2964903160264539dcae634abdd9a23ca8/slot_attention

The slot attention model was implemented using this repo: https://github.com/lucidrains/slot-attention

The task implemented here is that of object discovery on the clevr dataset.

The results are shown using tensorboard and the checkpoints are saved every epoch. In order to train the model use the train_clevr file, for other datasets the dataset will have to be replaced.
