## Description

This is the code for the paper [Neural Task Synthesis for Visual Programming, TMLR 2024](https://openreview.net/pdf?id=aYkYajcJDN).

## Structure
- ``data`` contains both synthetic data, used for training and evaluation, as well as GPT-4 and human-readable specifications used for the real-world task specification evaluation. It also contains the expert-designed codes used for the runtime evaluation.
- ``models`` contains pretrained models for code generation and task generation. The newly trained models will also be saved here.
- ``src`` contains the source code. It is structured as follows:
```
src/
├── codegen
│   ├── callbacks.py
│   ├── codegen.py
│   ├── codegraph.py
│   ├── code_sketches.py
│   ├── converter.py
│   ├── data.py
│   ├── decision_makers.py
│   ├── definitions.py
│   ├── __init__.py
│   ├── networks.py
│   ├── symast.py
│   ├── training.py
│   └── utils.py
├── codetask_scoring
│   ├── coverage.py
│   ├── deltadebugging.py
│   ├── execquality.py
│   ├── finalscore.py
│   ├── __init__.py
│   ├── shortestpath.py
│   ├── solvability.py
│   └── task_dissimilarity.py
├── emulator
│   ├── code.py
│   ├── executor.py
│   ├── fast_emulator.py
│   ├── __init__.py
│   ├── task.py
│   ├── tokens.py
│   └── world.py
├── evaluation
│   ├── full_eval.py
│   ├── full_gather.py
│   ├── __init__.py
│   ├── runtime_eval.py
│   ├── runtime_gather.py
│   ├── taskgen_eval.py
│   ├── taskgen_gather.py
│   └── utils.py
├── inference
│   ├── inference.py
│   ├── __init__.py
│   └── utils.py
├── symexecution
│   ├── decision_makers.py
│   ├── __init__.py
│   ├── post_processor.py
│   ├── README.md
│   ├── symworld.py
│   └── utils
│       ├── enums.py
│       ├── __init__.py
│       └── quadrants.py
├── taskgen
│   ├── callbacks.py
│   ├── data.py
│   ├── decision_makers.py
│   ├── feature_processors.py
│   ├── indexed_code.py
│   ├── __init__.py
│   ├── networks.py
│   ├── task_synthesizer.py
│   ├── training.py
│   └── vocabularies.py
└── utils
    ├── colors.py
    └── __init__.py
```
The scripts intended for usage (which shall be described in the next section) are located in the inference, taskgen, codegen and evaluation folders.

## Dependencies/Installation
The project is intended to be used with Python 3.9. We recommend using ``conda`` for package management. The following packages are required to run the code:
- ``tqdm``
- ``scipy``
- ``Python-Levenshtein``
- ``pytorch``

The first three can be installed using the provided ``requirements.txt`` file:
```shell
pip install -r requirements.txt
```

However, ``pytorch`` needs to be installed separately. Go to https://pytorch.org/ and select your system. The project was run with ``pytroch 2.1``, on ``Linux``, using ``conda``, in a ``Python 3.9`` environment, with ``CPU`` only as the compute platform. 


## Executing programs

We give examples of various use cases below.

### Run demo
This will run the demo for the specified specification. The demo will generate a task and its solution code.
```shell
python -m src.inference.inference [-h] 
                    [--algo_type {base,neural}]
                    [--task_decision_maker_path TASK_DECISION_MAKER_PATH]
                    [--code_decision_maker_path CODE_DECISION_MAKER_PATH]
                    [--spec_nb {0,1,2,3,4,5,6,7,8,9}]
                    [--maximum_blocks MAXIMUM_BLOCKS]
                    [--code_trials CODE_TRIALS] 
                    [--task_trials TASK_TRIALS]


```
Arguments:
- ``--algo_type``: algorithm type (base/neural), default=neural
- ``--task_decision_maker_path``: path to taskgen model, defaults to the model already present in the repository
- ``--code_decision_maker_path``: path to codegen model, defaults to the model already present in the repository
- ``--spec_nb``: the specification which is the input to the pipeline (0-9), default=8
- ``--maximum_blocks``: maximum number of blocks allowed in the generated code, default=9
- ``--code_trials``: number of code trials to run for the specification, default=5
- ``--task_trials``: number of task trials to run for one code, default=100

### Run taskgen training
This will train the taskgen model and save it in ``models/[domain]/taskgen/``. An example model is already present in the repository, so this step is not necessary to run the demo.
```shell
python -m src.taskgen.training [-h]
                      [--seed SEED] 
                      [--domain DOMAIN] 
                      [--learning_rate LEARNING_RATE] 
                      [--batch_size BATCH_SIZE]
                      [--epochs EPOCHS] 
                      [--temperature TEMPERATURE]
                      [--cnn_layer_sizes CNN_LAYER_SIZES [CNN_LAYER_SIZES ...]] 
                      [--pooling_layer_sizes POOLING_LAYER_SIZES [POOLING_LAYER_SIZES ...]] 
                      [--grid_stack GRID_STACK [GRID_STACK ...]]
                      [--decision_stack DECISION_STACK [DECISION_STACK ...]]
```
Arguments:
- ``--seed``: manual seed setting for reproducibility, default=0
- ``--domain``: domain type (hoc/karel), default=hoc
- ``--learning_rate``: learning rate during training, default=1e-4
- ``--batch_size``: batch size during training, default=32
- ``--epochs``: number of epochs to train for, default=500
- ``--temperature``: temperature applied during training, before softmax, default=1.0
- ``--cnn_layer_sizes``: list of layer depths for CNN, default=[64, 64, 64]
- ``--pooling_layer_sizes``: list of sizes for pooling, default=[2, 2, 2]
- ``--grid_stack``: list of fully connected layer sizes applied after CNN, default=[1024, 512, 256, 128, 32]
- ``--decision_stack``: list of fully connected layer sizes applied after concatenating the grid embedding and the code features, default=[8]

### Run codegen training
This will train the codegen model and save it in ``models/[domain]/codegen/``. An example model is already present in the repository, so this step is not necessary to run the demo.
```shell
python -m src.codegen.training [-h]
                      [--seed SEED] 
                      [--domain DOMAIN]
                      [--epochs EPOCHS] 
                      [--learning_rate LEARNING_RATE] 
                      [--batch_size BATCH_SIZE] 
                      [--embedding_size EMBEDDING_SIZE] 
                      [--num_layers NUM_LAYERS] 
                      [--hidden_size HIDDEN_SIZE]

```
Arguments:
- ``--seed``: manual seed setting for reproducibility, default=0
- ``--domain``: domain type (hoc/karel), default=hoc
- ``--learning_rate``: learning rate during training, default=5*1e-4
- ``--batch_size``: batch size during training, default=32
- ``--epochs``: number of epochs to train for, default=100
- ``--embedding_size``: size of embedding for code tokens, default=256
- ``--num_layers``: number of layers in the LSTM, default=2
- ``--hidden_size``: size of hidden state in the LSTM, default=256

### Run taskgen evaluation
This will evaluate the taskgen model on the test set and save the results in ``results/[domain]/seed_[seed]/``.
```shell
python -m src.evaluation.taskgen_eval [-h]
                      [--seed SEED]
                      [--domain DOMAIN] 
                      [--algo_type {base,neural}] 
                      [--decision_maker_path DECISION_MAKER_PATH]
                      [--task_trials TASK_TRIALS]
                      [--nb_processes NB_PROCESSES] 
```
Arguments:
- ``--seed``: manual seed setting for reproducibility, default=0
- ``--domain``: domain type (hoc/karel), default=hoc
- ``--algo_type``: algorithm type (base/neural), default=neural
- ``--decision_maker_path``: path to taskgen model, defaults to the model already present in the repository
- ``--task_trials``: number of task trials to run for one code, default=100
- ``--nb_processes``: number of processes to use for parallel evaluation, default=10

### Taskgen evaluation results gathering
This will gather the results from the evaluation and save them in ``results/[domain]``.
```shell
python -m src.evaluation.taskgen_gather [-h]
                      [--domain DOMAIN] 
                      [--seeds SEEDS [SEEDS ...]]
```
Arguments:
- ``--domain``: domain type (hoc/karel), default=hoc
- ``--seeds``: list of seeds to gather results for, default=[0]

### Run full evaluation
This will evaluate the full architecture on the test set and save the results in ``results/[domain]/seed_[seed]/``.
```shell
python -m src.evaluation.full_eval [-h]
                      [--seed SEED] 
                      [--domain DOMAIN] 
                      [--algo_type {base,neural}] 
                      [--task_decision_maker_path TASK_DECISION_MAKER_PATH] 
                      [--code_decision_maker_path CODE_DECISION_MAKER_PATH] 
                      [--code_trials CODE_TRIALS] 
                      [--task_trials TASK_TRIALS] 
                      [--nb_processes NB_PROCESSES]
```
Arguments:
- ``--seed``: manual seed setting for reproducibility, default=0
- ``--domain``: domain type (hoc/karel), default=hoc
- ``--algo_type``: algorithm type (base/neural), default=neural
- ``--task_decision_maker_path``: path to taskgen model, defaults to the model already present in the repository
- ``--code_decision_maker_path``: path to codegen model, defaults to the model already present in the repository
- ``--code_trials``: number of code trials to run for one specification, default=5
- ``--task_trials``: number of task trials to run for one code, default=100
- ``--nb_processes``: number of processes to use for parallel evaluation, default=10

### Full architecture evaluation results gathering
This will gather the results from the evaluation and save them in ``results/[domain]``.
```shell
python -m src.evaluation.full_gather [-h]
                      [--domain DOMAIN] 
                      [--seeds SEEDS [SEEDS ...]]
```
Arguments:
- ``--domain``: domain type (hoc/karel), default=hoc
- ``--seeds``: list of seeds to gather results for, default=[0]

### Run runtime evaluation
This will evaluate the runtime and rollouts of the taskgen model on the codes selected from ``data/real-world/expert_codes.json``  (HoC:Maze8, HoC:Maze9, HoC:Maze13, HoC:Maze18, Karel:OurFirst, Karel:Diagonal).
```shell
python -m src.evaluation.runtime_eval [-h]
                      [--algo_type {base,neural}]
                      [--decision_maker_path DECISION_MAKER_PATH] 
                      [--percentage PERCENTAGE] 
```
Arguments:
- ``--algo_type``: algorithm type (base/neural), default=neural
- ``--decision_maker_path``: path to taskgen model, defaults to the models already present in the repository, depending on the domain
- ``--percentage``: percentage of the expert-designed task score the generation process should achieve, default=99

### Runtime evaluation results gathering
This will compute the average and standard error of the previously computed runtime and rollouts of the taskgen model.
```shell
python -m src.evaluation.runtime_gather [-h]
                      [--algo_type {base,neural}]
                      [--percentage PERCENTAGE]
```
Arguments:
- ``--algo_type``: algorithm type (base/neural), default=neural
- ``--percentage``: percentage of the expert-designed task score the average should be computed for, default=99

## Citation
```
@article{
    padurean2024neural,
    author  =   {Victor-Alexandru P\u{a}durean and Georgios Tzannetos and Adish Singla},
    title   =   {{N}eural {T}ask {S}ynthesis for {V}isual {P}rogramming},
    journal =   {Transactions of Machine Learning Research (TMLR)},
    year    =   {2024},
}
```