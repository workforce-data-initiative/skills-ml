import tensorflow as tf

from data_loader import DataGenerator
from models.model import ExampleModel
from trainers.trainer import ExampleTrainer
from logger import Logger
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def run():
    # capture the config path from the run arguments
    # then process the json configration file
    args = get_args()
    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create instance of the model you want
    model = ExampleModel(config)
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    run()
