# PDAM: A neural network based instant TSP solver.

Progressive Distillation Attention Model (PDAM) is a neural network based instant TSP solver, which is able to solve a TSP instance within several millon seconds. 

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Quick start

For training TSP instances with 20 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20_rollout'
```

## Usage

### Generating data

Training data is generated on the fly. To generate validation and test data (same as used in the paper) for all problems:
```bash
python generate_data.py --problem all --name validation --seed 4321
python generate_data.py --problem all --name test --seed 1234
```

### Training

For training TSP instances with 20 nodes and using rollout as REINFORCE baseline and using the generated validation set:
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20_rollout' --val_dataset data/tsp/tsp20_validation_seed4321.pkl
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --graph_size 100 --load_path pretrained/tsp_100/epoch-99.pt
```

The `--load_path` option can also be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --graph_size 20 --load_path 'outputs/tsp_20/tsp20_rollout_{datetime}/epoch-0.pt'
```

The `--resume` option can be used instead of the `--load_path` option, which will try to resume the run, e.g. load additionally the baseline state, set the current epoch/step counter and set the random number generator state.

### Progressive Distillation

You can set `--progressive_distillation` parameter to enable progressive distillation. `--beta` is the weight of progressive distillation. `--local_size` is local window size ratio, and `--beta_decay` is every epoch beta will decay.
```bash
python run.py --graph_size 20 --progressive_distillation  --local_size 0.4 --beta_decay 0.95 --load_path 'outputs/tsp_20/tsp20_rollout_{datetime}/epoch-0.pt'
```

### Evaluation
To evaluate a model, you can add the `--eval-only` flag to `run.py`, or use `eval.py`, which will additionally measure timing and save the results:
```bash
python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy greedy
```
If the epoch is not specified, by default the last one in the folder will be used.

#### Sampling
To report the best of 1280 sampled solutions, use
```bash
python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy sample --width 1280 --eval_batch_size 1
```
Beam Search (not in the paper) is also recently added and can be used using `--decode_strategy bs --width {beam_size}`.

### Other options and help
```bash
python run.py -h
python eval.py -h
```
