"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSet':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params

print('--Parsing Config File')
params = process_config('config.cfg')

print('--Creating Dataset')
dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'])
dataset._create_train_table()
dataset._randomize()
dataset._create_sets()

model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= False, modif=False)
model.generate_model()
model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None)

