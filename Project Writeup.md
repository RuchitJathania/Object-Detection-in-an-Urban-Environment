__Note: I have completed the project through Udacity's VM workspace, so many local work specific tasks and version control weren't looked into. I did not have to implement the create_splits.py program since the data splits were taken care of in the VM workspace but have completed the function anyways.__ 
# Project Writeup

## Project Overview
This project introduces students to the ML Object Detection workflow utilizing the skills of data analysis, visualization, data augmentation, and running a Convolution Neural Network with FFNN to create a model. Images with groundtruths from Waymo were used as input data, model training was done trough the use of TensorFlow Object Detection API, and tensorboard was used to visualize the model's metrics and performance. The goal was to familiarize students with the Object Detection workflow and successfully train and evaluate the ML model with certain hyper-parameter changes of choice.

## Set Up
*Note: Instructions to train and run the model with given* `.config` *files*

First, get the files from the [Github Repository](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment.git)

Then build the Docker container and image with the dockerfile and instructions in the `build` directory. This requires a Nvidia-GPU and either docker or nvidia-docker. 
Once all of the dependencies are installed and the project is ready to be ran through the docker container continue to the next steps. 
Most of the command line arguments need to run from the directory of the project.

You need to download the data to be used for the model by using the `dowlaod_process.py` code by running:
```bash
  python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```
Then you need to create the splits for the data by running `create_splits.py` by running this in your terminal:
```bash
  python create_splits.py --data-dir {Your directory for processed_file_location}
```
Then you have to download the [pretrained SSD Resnet 50 640x640 model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz), extract and put it in `experiments/pretrained_model/`

Then to train the model, run this in your terminal:
```bash
  python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path={Path of config file you want to use}
```

Then to evaluate the model run the following but change the `experiments/reference/checkpoint` file's first line to `ckpt-1` and increase each time evaluation is done:
```bash
  python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path={Path of config file you want to used to train} --checkpoint_dir=experiments/reference/
```

Finally to visualize the performance metrics through tensorflow, run the following in your terminal:
```bash
  python -m tensorboard.main --logdir experiments/reference/
```
## Dataset Analysis

![](EDA/edaimg.png)

![](EDA/Screenshot%202022-06-24%20154321.png)

![](EDA/Screenshot%202022-06-24%20154346.png)

![](EDA/Screenshot%202022-06-24%20154418.png)


![](EDA/Screenshot%202022-06-24%20154431.png)

![](EDA/Screenshot%202022-06-24%20154502.png)

![](EDA/segment-10017090168044687777_6.png)

![](EDA/segment-10023947602400723454_1.png)

![](EDA/segment-1005081002024129653_53.png)

![](EDA/segment-10107710434105775874_7.png)

![](EDA/segment-1022527355599519580_48.png)

![](EDA/segment-10235335145367115211_5.png)

![](EDA/segment-10241508783381919015_2.png)

![](EDA/segment-10327752107000040525_1.png)

![](EDA/segment-10723911392655396041_8.png)

![](EDA/segment-11004685739714500220_2.png)

![](EDA/segment-11070802577416161387_7.png)

![](EDA/segment-11113047206980595400_2.png)

![](EDA/segment-11219370372259322863_5.png)

![](EDA/segment-11355519273066561009_5.png)

![](EDA/segment-11674150664140226235_6.png)

![](EDA/segment-11839652018869852123_2.png)

![](EDA/segment-11847506886204460250_1.png)

![](EDA/segment-1191788760630624072_38.png)

These images show a wide range of domains and variation in the training images.
* The traffic density is highly variable
* The weather conditions are highly variable(fog, rain, overcast)
* Certain images are from night time
* The relative size of vehicles is highly variable
* The angle of vehicle with respect to the camera is variable, producing different features for the camera
* General environments differ, some are city streets, others highways, some suburban
* Some occlusion in some objects

_Note: The classes of object are color coded with cars being red, pedestrians green, cyclists in yellow_
## Training

### Reference Experiment

### Custom Experiment/Model Improvements