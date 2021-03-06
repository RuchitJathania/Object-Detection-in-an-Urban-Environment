# Object-Detection-in-an-Urban-Environment
Udacity Self-Driving Car Engineering Course Computer Vision Project

## Set Up (Also in Project Writeup)
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
