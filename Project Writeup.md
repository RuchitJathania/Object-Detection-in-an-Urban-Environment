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
![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/Screenshot%202022-06-24%20154321.png)

![]()
![]()
![]()


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

![](experiment0_ref/refEvalLoss.png)

![](experiment0_ref/refTrainStep.png)

![](experiment0_ref/refEvalPrecision.png)

![](experiment0_ref/refEvalRecall.png)

Overall, the reference model did not perform too well. The parameters were being updated well and optimized until a 1600 steps when the learning rate was around 0.015 but then flattened out to a loss metric of around 4.5. Of course, as expected the evaluation loss was slightly higher than the training loss since none of the evaluation images were the same as the training, getting rid of the bias of the same data being fed to the model.
### Custom Experiment/Model Improvements