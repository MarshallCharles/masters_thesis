# Robust Model Compression for Limited Resource Hardware

## Contents
- [Notebook](./Interpreting_models.ipynb) for CNN interpretability tooling. 
- [Training Module](./train_model.py) for various models.
- [Testing Module](./test_model.py) for evaluating performance of models. 
- [Model](./models) directory in which I provide many pretrained models.
- [Data](./data/) directory in which CIFAR10 & CIFAR100 will be downloaded.
- [Dependencies](./requirements.txt)

## Running Experiments

#### Requires
- A python distribution >= 3.6

#### Setting up docker (not required)
- If you do not have docker installed, please refer to the [official documentation](https://www.docker.com/)
- More details on docking this code upcoming!

#### Setting up a virtual environment with python (recommended if not using docker)
- Make a venv directory at `./env` as follows: `python -m venv ./env`
- Then, `source ./env/bin/activate` to start the virtual environment.
- Now that the venv is activated, running `pip install -r requirements.txt` will install all the necessary dependencies to the vm. 

## Testing Models
#### Due to the demand for running many tests in a streamlined fashion, I provide two options:
1. Running a single test is as simple as running the script `test_model.py [PATH_TO_MODEL]` where `PATH_TO_MODEL` is the path from the working directory to a model you would like to test. 
	* All of the additional (optional) flags can be used in this fashion and are available below or with the `-h` flag; `python [train/test]_model.py -h`. 
	* If running an experiment in this fashion, use `--[ARGUMENT] [VALUE]` to secify argument values at runtime.

2. Running multiple tests at once can be done with a config file. This file, preferably without extension, or `.txt`, is passed at runtime; `test_model.py [PATH_TO_CONFIG]`. This file can be placed anywhere and needs only read privileges from the working directory. 
	* On each line of this file there should be parameters for an experiment starting with the model path from the working directory, and the arguments; `[PATH_TO_MODEL] [ARGUMENT 1] [ARGUMENT 1 VALUE] `. Note that the double dashes are not required here.
	* At runtime, the default parameters defined in `test_model.py` are used. A parameter is only ever overriden when its value is specified in a line of the config file. In this way, running the same experiment on many models requires putting the arguments only on the first line, and all other lines will contain just the paths to the subsequent models.  

#### A suggestion
* By default, the test script will check if there are GPUs available on the hardware you are using. I recommend always omitting the following 3 arguments: `gpu`, `ngpu`, `seed`.

#### Arguments for `test_model.py` or any config file
* config (or model): test config file, or a .pth file. This is the only required argument, and should be used first when running any test. There is no need to specify that this argument is being passed, simply pass a path as the argument in first position.
* `architecture [resnet/mobilenet/shufflenet]`: architecture of the model specified for this test. Shufflenet currently not supported. 
* `effnet_alpha [FLOAT]`: alpha multiplier if using mobilenet or shufflenet.
* `shufflenet_groups [INT]`: amount of groups if using shufflenet. 
* `help`: same as `-h`, shows the help message and exits.
* `log [PATH]`: log file to which test results will be appended.
* `verbose [True/False]`: print verbose.
* `gpu [INT]`: index of GPUs to use.
* `ngpu [INT]`: number of GPUs to use.
* `seed [INT]`: random seed.
* `dataset [cifar10/cifar100]`: dataset to use (2 choices)
* `acc_test [True/False]`: run inference on test set and report accuracy and inference speed.
* `adv_acc_test [True/False]`: run inference on adversarial test set and report accuracy and inference speed.
* `atk_algo [pgd/fgsm]`: attack for the adversarial accuracy test.
* `epsilon [FLOAT]`: epsilon value for adversarial gradient evaluations.
* `attack_iterations [INT]`: max number of gradient evaluations if using an iterative attack.
* `test_set_iteration_cap [INT]`: max number of images to pull from the test set.
* `acc_eps_test [True/False]`: run the elbow test for this model (warning, this test is very time consuming and should be done only on GPU hardware)

## Training new models
Training a model from scratch, or using another model as a base can be done with the script `train_model.py`. All arguments for this script are decscribed below, followed by explicit instructions to reproduce experiments.

Note that double dashes are needed when running the training script, and no config file is used. Declare arguments as follows: `--[ARGUMENT] [ARGUMENT VALUE]`. For arguments with `[True/False]` options, simply declaring the argument will enable it as `True`, while omitting it will keep its value set to `False`.

#### A suggestion
* Similar to the testing script, I recommend the following arguments always be omitted: `gpu`, `ngpu`, `seed`. Additionally, unless deviating entirely from the examples, I recommend also leaving `weight_decay` and `learning_rate` unchanged.

#### Arguments for `train_model.py`
* `architecture [resnet/mobilenet/shufflenet]`: Model architecture to be used.
* `effnet_alpha [FLOAT]`: Alpha value if using mobilenet architecture (default 1.0)
* `shufflenet_groups [INT]`: Groups if using shufflenet architecture.
* `raw_train [True/False]`: Train a model from scratch. If specified, no need to use the `--model` flag.
* `model [PATH_TO_MODEL]`: Path to a .pth file to use as a base if not using `--raw_train`.
* `batch_size [INT]`: Size of batches to use in each epoch.
* `epochs [INT]`: Amount of epochs to train.
* `weight_decay [FLOAT]`: Rate of weight decay.
* `learning_rate [FLOAT]`: Initial learning rate.
* `decreasing_lr [INT,INT, ...]`: List of comma separated integers corresponding to epochs in which learning rate will be decreased. Not applicable if using a different learning rate scheme. 
* `gpu [INT]`: index of GPUs to use.
* `ngpu [INT]`: number of GPUs to use.
* `seed [INT]`: random seed.
* `test_interval [INT]`: How many epochs to wait before a validation test.
* `save_dir [PATH]`: Path to the directory in which logs will be saved.
* `dataset [cifar10/cifar100]`: Which dataset to use.
* `log_interval [INT]`: Amount of batches between writing to log and stdout (if verbose).
* `ada_train_attack [True/False]`: Use adaptive scheme on top of selected attack.
* `maximum_epsilon [FLOAT]`: Maximum perturbation bound to reach if using `ada_train_attack`.
* `defense_algo [pgd/fgsm]`: Algorithm to use for defense.
* `defense_iterations [INT]`: Amount of iterations if using an adaptive defense algorithm.
* `attack_algo [pgd/fgsm]`: Algorithm to use for validation tests.
* `attack_radius [FLOAT]`: Perturbation bound to use for validation tests.
* `attack_iterations [INT]`: Amount of iterations if using an adaptive attack algorithm.
* `prune_algo [l0proj/adal0proj]`: Pruning algorithm to use.
* `prune_ratio [FLOAT]`: Number in [0..1] corresponding to the compression ratio to reach.
* `prune_interval [INT]`: Number of epochs to wait between puning attempts.
* `verbose [True/False]`: Print verbose.

#### Examples
Below I go through the exact steps to reporoduce some of my experiments.

##### Training a dense resnet model from scratch
The following command can be used, which uses most parameters as default
* `python ./train_model.py --raw_train --test_interval 10 --dataset cifar10 --verbose --epochs 200 --log_interval 25 --save_dir logs/ --architecture resnet`

##### Pruning the dense model using an adaptive pruning scheme and defense method.


