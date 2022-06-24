import argparse
import glob
import os
import random
import shutil

import numpy as np

# from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    file_list = glob.glob(source+"\*.tfrecord")
    train_list = random.sample(file_list, 90)
    eval_list = random.sample(file_list,10)
    shutil.rmtree(destination+r"\train")
    shutil.rmtree(destination+r"\eval")
    shutil.rmtree(destination+r"\test")
    os.mkdir(destination+r"\train")
    os.mkdir(destination+r"\eval")
    os.mkdir(destination+r"\test")
    for filepath in train_list:
        shutil.copy(filepath,destination+r"\train")
    for filepath in eval_list:
        shutil.copy(filepath,destination+r"\eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=False,
                        help='source data directory', default=r'C:\OnlineCourses\SelfDrivingCarEngineer\ObjectDetectioninUrbanEnvironment\data\all')
    parser.add_argument('--destination', required=False,
                        help='destination data directory',default=r'C:\OnlineCourses\SelfDrivingCarEngineer\ObjectDetectioninUrbanEnvironment\data\newdata')
    args = parser.parse_args()

    # logger = get_module_logger(__name__)
    # logger.info('Creating splits...')
    split(args.source, args.destination)