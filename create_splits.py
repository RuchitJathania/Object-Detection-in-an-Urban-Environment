import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger

def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in
    the same directory. This folder should be named train, val and test. This function randomly
    takes 87% of the tfrecords from processed data folder and puts them in a new training folder,
    10% into the evaluation folder and the rest in the test folder. This code also deletes any
    files in the train and eval folders before it randomly splits the data to ensure no
    duplicates are added when re-running the code args: - source [str]: source data directory,
    contains the processed tf records - destination [str]: destination data directory, contains 3
    sub folders: train / val / test
    """
    # TODO: Implement function
    file_list = glob.glob(source + r"\*.tfrecord")
    num_tfrecords = float(len(file_list))
    num_train = int(0.87 * num_tfrecords)
    num_eval = int(0.1 * num_tfrecords)
    num_test = int(num_tfrecords) - num_train - num_eval
    print(num_tfrecords, num_train, num_eval, num_test)
    train_list = random.sample(file_list, num_train)
    eval_list = random.sample(file_list, num_eval)
    test_list = random.sample(file_list, num_test)
    shutil.rmtree(destination + r"\train")
    shutil.rmtree(destination + r"\eval")
    shutil.rmtree(destination + r"\test")
    os.mkdir(destination + r"\train")
    os.mkdir(destination + r"\eval")
    os.mkdir(destination + r"\test")
    for filepath in train_list:
        shutil.copy(filepath, destination + r"\train")
    for filepath in eval_list:
        shutil.copy(filepath, destination + r"\eval")
    for filepath in test_list:
        shutil.copy(filepath, destination + r"\test")


# Default arguments for parser are for my own local directory, please supply your own when running
# the code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=False,
                        help='source data directory',
                        default=r'C:\OnlineCourses\SelfDrivingCarEngineer\ObjectDetectioninUrbanEnvironment\data\all')
    parser.add_argument('--destination', required=False,
                        help='destination data directory',
                        default=r'C:\OnlineCourses\SelfDrivingCarEngineer\ObjectDetectioninUrbanEnvironment\data\newdata')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
