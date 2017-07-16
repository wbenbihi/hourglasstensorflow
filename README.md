# Stacked Hourglass model : TensorFlow implementation
Tensorflow implementation of Stacked Hourglass Networks for Human Pose Estimation by A.Newell et al.
## Based on
[Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937) -- A.Newell et al.
## Status
This is a WIP repo

## CONFIG FILE
A 'config.cgf' is present in the directory.
	It contains all the variables needed to tweak the model.
	
	training_txt_file : Path to TEXT file containing information about images
	img_directory : Path to folder containing images
	img_size : Size of input Image /!\ DO NOT CHANGE THIS PARAMETER (256 default value)
	hm_size : Size of output heatMap /!\ DO NOT CHANGE THIS PARAMETER (64 default value)
	num_joints : Number of joints considered
	joint_list: List of joint name
	name : Name of trained model
	nFeats: Number of Features/Channels in the convolution layers (256 / 512 are good but you can set whatever you need )
	nStacks: Number of Stacks (4 to make the faster prediction, 8 stacks are used in the paper)
	nModules : NOT USED
	nLow : Number of downsampling in one stack (default: 4 => dim 64->4)
	dropout_rate : Percentage of neurons deactivated at the end of Hourglass Module (Used for training)
	batch_size : Size of training batch (8/16/32 are good values depending on your hardware)
	nEpochs : Number of training epochs
	epoch_size : Iteration in a single epoch
	learning_rate: Starting Learning Rate
	learning_rate_decay: Decay applied to learning rate (in ]0,1], 0 not included), set to 1 if you don't want decay learning rate. (Usually, keep decay between 0.9 and 0.99)
	decay_step : Step to apply decay to learning rate
	valid_iteration : Number of prediction made on validation set after one epoch (valid_iteration >= 1)
	log_dir_test : Directory to Test Log file
	log_dir_train : Directory to Train Log file
	saver_step : Step to write in train log files (saver_step < epoch_size)
	saver_directory: Directory to save trained Model
