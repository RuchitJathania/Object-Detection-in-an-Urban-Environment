__Note: I have completed the project through Udacity's VM workspace, so many local work specific tasks and version control weren't looked into. I did not have to implement the create_splits.py program since the data splits were taken care of in the VM workspace but have completed the function anyways. Also, the images and visaulizations in this md file are through github links. To view the images if they do not load here correctly: go to `Object-Detection-in-an-Urban-Environment/DataVisualization/` and `Object-Detection-in-an-Urban-Environment/experiments/experiment0/` in the repository for each experiment(0,1,2)__ 
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

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/Screenshot%202022-06-24%20154346.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/Screenshot%202022-06-24%20154418.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/Screenshot%202022-06-24%20154431.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/Screenshot%202022-06-24%20154502.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/edaimg.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10017090168044687777_6.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10023947602400723454_1.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-1005081002024129653_53.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10107710434105775874_7.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-1022527355599519580_48.png)


![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10235335145367115211_5.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10235335145367115211_5.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10241508783381919015_2.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10327752107000040525_1.png)


![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-10723911392655396041_8.png)


![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11004685739714500220_2.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11070802577416161387_7.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11113047206980595400_2.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11219370372259322863_5.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11355519273066561009_5.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11674150664140226235_6.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11839652018869852123_2.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-11847506886204460250_1.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploratoryDataAnalysis/segment-1191788760630624072_38.png)

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

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment0/refEvalLoss.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment0/refTrainStep.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment0/refEvalPrecision.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment0/refEvalRecall.png)

Default train config hyper-parameters:
```text
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.04
          total_steps: 2500
          warmup_learning_rate: 0.013333
          warmup_steps: 200
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
Overall, the reference model did not perform too well. The parameters were being updated well and optimized until a 1600 steps when the learning rate was around 0.015 but then flattened out to a loss metric of around 4.5. Of course, as expected the evaluation loss was slightly higher than the training loss since none of the evaluation images were the same as the training, getting rid of the bias of the same data being fed to the model.
### Custom Experiment/Model Improvements
#### Experiment 1
Config file changes:
```text
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options{
  	random_distort_color{}
  }
  data_augmentation_options{
  	random_adjust_contrast{
    	min_delta:1.0
        max_delta:1.20
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
```

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment1/trainLoss.png)


![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment1/learning_rate.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment1/precision.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment1/recall.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment1/ex1.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment1/ex2.png)

The color distortion augmentation was added to make the model less prone to training features/patterns based on certain colors and rather focus on the underlying features of the objects such as shapes and other patterns. The random contrast adjustment was added to further exemplify those shapes and patterns of the objects since many images were blurry due to fog or weather being sub-optimal. 

The performance overall was slightly better than the reference but the training losses would spike every 200 steps or so probably due to unlucky batch data and optimization parameters not fitting well. 

#### Experiment 2
Config file changes from default:

```text
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options{
  	random_distort_color{}
  }
  data_augmentation_options{
  	random_adjust_contrast{
    	min_delta:1.1
        max_delta:1.20
    }
  }
  data_augmentation_options{
  	random_adjust_saturation{
    	min_delta:1.1
        max_delta:1.3
    }
  }
  data_augmentation_options{
	random_rgb_to_gray{
	probability:0.30
	}  
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    adam_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.04
          total_steps: 2500
          warmup_learning_rate: 0.0135
          warmup_steps: 200
        }
      }
    }
    use_moving_average: false
  }
```

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment2/loss.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/experiments/experiment2/learningRate.png)


![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment2/Screenshot%202022-06-24%20155500.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment2/1.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment2/3.png)


![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment2/7.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment2/8.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment2/5.png)

![](https://github.com/RuchitJathania/Object-Detection-in-an-Urban-Environment/blob/main/DataVisualization/ExploringAugmentations/Experiment2/Screenshot%202022-06-24%20155500.png)

To further decrease reliance on certain color to detect objects and further exaggerate the base features of groundtruth objects, `random_rgb_to_gray` augmentation and a random saturation enhancement was added. Furthermore, since the reference model plateaued in loss decrease, the optimizer was changed to the more complex Adam optimizer in hopes that any local minima were avoided and optimal changes were made to weights. 

Unfortunately, although the loss curve was more stable and smoother, the overall loss at the end of training was very high at around 5.0. 

Note: The config file changes were researched and found in the 
[preprocessing proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor proto) and the 
[optimizer proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/optimizer.proto)


