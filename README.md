# RelDDPM

## Introduction

This is the source code of the paper **Controllable Tabular Data Synthesis Using Diffusion Models**

## Quick Start
### Environment Setup
Before running the code, please make sure your Python version is above **3.7**.
We recommend running the code under a virtual environment:
```sh
conda create -n relddpm_env python=3.8
conda activate relddpm_env
```

Then install the necessary packages by :
```sh
pip install -r requirements.txt
```

### Code Structure
```sh
|-- datasets
    |-- minority_class_oversampling # datasets used in minority class oversampling task
    |-- missing_tuple_completion # datasets used in missing tuple completion task
|-- ddpm # the denoise diffusion probabilistic model package
|-- lib_completion # the library used in missing tuple completion task 
|-- lib_oversampling # the library used in minority class oversampling task 
|-- data_utils.py # the class to preprocess the dataset
|-- eval_utils.py # the class to evaluate
|-- eval.py # code of the evaluation
|-- main.py # main code
```

### Run
#### Minority Class Oversampling.
Run the code to generate synthetic data for minority class oversampling with the following command:
```sh
python main.py --task-name=oversampling --dataset-name=[dataset] --device=[GPU id] --save-name=[output file]
```
The parameter "dataset" should be "default", "shoppers" or "weatherAUS".

For example:
```sh
python main.py --task-name=oversampling --dataset-name=default --device=0 --save-name=default_output
```
#### Minority Class Oversampling.
Run the code to generate synthetic data for missing tuple completion with the following command:
```sh
python main.py --task-name=completion --dataset-name=[dataset] --device=[GPU id] --save-name=[output file]
```
The parameter "dataset" should be "heart", "airbnb" or "imdb".

### Evaluation
Run the code to evaluate the results of the minority class oversampling/missing tuple completion with the following command:
```sh
python eval.py --task-name=[task name] --dataset-name=[dataset] --device=[GPU id] --save-name=[output file]
```

To evaluate the performance of *minority class oversampling* on *default* dataset, assume the synthetic results are saved in *default_output*, the command is:
```sh
python eval.py --task-name=oversampling --dataset-name=default --device=0 --save-name=default_output
```


