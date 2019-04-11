#SG-One-pytorch 

 [Paper: SG-One: Similarity Guidance Network for One-Shot Semantic Segmentation](https://arxiv.org/abs/1810.09091)

this project is a modified version of author's original codes.I managed to run the code successfully and add some comments. Meanwhile i change the way of loading pretrained model. For more detials, please reference to the [authors's repo](https://github.com/xiaomengyc/SG-One)

Please note this project have not achieved the performance in authors' paper.

## Setup the Environment

`recommand to install the anaconda`

- pytorch 0.4.0+
- python3 +
- EasyDict


## Setup the project
you have to change the training parameters in `config.py`

in `DataCapsule/db_config.py` you can set the information about the pascal voc datatsets. For example, change the __C.PASCAL_PATH to your own path.


## Start to train
run `train_frame.py` and start to train. you can give the args like --group=0 to train the model for pascal-0

## Start to test
run `test_frame.py` you can give the args like --group=0 to train the model for pascal-0, --restore_step=10000 to restore the saved parameters.

## To-Do list
- rewrite  `test_frame_5shot_max.py` and `test_frame_5shot_avg.py`
- try to get the similar result in the paper
- rewrite the whole part of DataCapsule