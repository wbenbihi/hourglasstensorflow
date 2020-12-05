import pandas as pd
from config import CFG
import tensorflow as tf
from .functions import tf_parse_dataset, tf_preprocess


class HPEDatasetGenerator:

    def __init__(self, csv_filepath, image_folder, csv_args={}):
        self.csv_filepath = csv_filepath
        self.image_folder = image_folder
        self.csv_args = csv_args
        self.dataset_df = self._load_csv()
        self.dataset = None

    def _load_csv(self):
        return pd.read_csv(self.csv_filepath, **self.csv_args)

    def init_pipe(self):
        self.pipe_df = self.dataset_df.copy()
        return self

    def split_train(self):
        self.pipe_df = self.pipe_df.query("is_train == 1")
        return self
    
    def split_test(self):
        self.pipe_df = self.pipe_df.query("is_train != 1")
        return self

    def dummy_test(self):
        self.is_train = self.pipe_df.is_train.tolist()
        self.filenames = (self.image_folder + '/' + self.pipe_df.filter(regex="image")).image.tolist()
        self.coordinates = self.pipe_df.filter(regex="joint_[0-9]*_(x|y)").values.reshape((-1, CFG.default.DATA.njoints, 2))
        return self

    def get_dataset(self):
        filenames = (self.image_folder + '/' + self.pipe_df.filter(regex="image")).image.tolist()
        coordinates = self.pipe_df.filter(regex="joint_[0-9]*_(x|y)").values.reshape((-1, CFG.default.DATA.njoints, 2))

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (filenames, coordinates)
        )
        return self

    def preprocess_dataset(self):
        self.dataset = self.dataset.map(tf_parse_dataset).map(tf_preprocess)
        return self

    def apply_random_rotation(self):
        self.dataset = self.dataset.map(RANDOM_ROTATION_FUNCTION)
        return self

    def shuffle(self, n):
        self.dataset = self.dataset.shuffle(n)
        return self

    def batch(self, n):
        self.dataset = self.dataset.batch(n)
        return self

    def take(self, n):
        self.dataset = self.dataset.take(n)
        return self