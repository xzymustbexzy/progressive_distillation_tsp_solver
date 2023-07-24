#!/usr/bin/env python
import time
import os
import json
import pprint as pp
import math
import re
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger
from tqdm import tqdm

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from nets.attention_model import set_decode_type
from utils import torch_load_cpu, load_problem, move_to


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    if opts.decode_strategy == 'bs':
        cost = beam_search(model, dataset, opts)
    elif opts.decode_strategy == 'greedy':
        cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def beam_search(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "bs")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model.beam_search(move_to(bat, opts.device), beam_size=opts.width)
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def load_small_model(problem, model_class, opts, path):
    small_model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        action_mask=opts.action_mask,
        k=opts.k
    )
    load_data_small = torch_load_cpu(path)
    small_model.load_state_dict({**small_model.state_dict(), **load_data_small.get('model', {})})
    small_model.eval()
    small_model = small_model.to(opts.device)
    return small_model


def run_local_prob(small_model, input, state):
    with torch.no_grad():
        embeddings, _ = small_model.embedder(small_model._init_embed(input))
        fixed = small_model._precompute(embeddings)
        log_p, mask, origin_log_logits = small_model._get_log_p(fixed, state)
    return log_p


def task_size_linear(s, t, N, epoch):
    return min(t, s+int((t-s)/N*epoch))


def task_size_logarithmic(s, t, N, epoch):
    return min(t, int((t-s)//math.log(N+1)*math.log(epoch+1)+s))


def search_best_model(total_size, radius, save_dir):
    estimate_size = (radius * radius) * total_size
    filenames = os.listdir(save_dir)
    history_info = []
    for filename in filenames:
        if filename.endswidth('.pt'):
            m = re.match(r'epoch-(\d+)-size-(\d+).pt', filename)
            history_info.append([int(m.group(1)), int(m.group(2))])
    min_dist = 1e9
    best_epoch = -1
    for epoch, size in history_info:
        if abs(size - estimate_size) < min_dist:
            min_dist = abs(size - estimate_size)
            best_epoch = epoch
    return best_epoch, min_dist

def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        action_mask=opts.action_mask,
        k=opts.k,
        progressive_distillation=opts.progressive_distillation
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    load_small_model_ = partial(load_small_model, problem, model_class, opts)

    if opts.eval_only:
        start = time.time()
        validate(model, val_dataset, opts)
        end = time.time()
        print(f'use time: {end - start} s')
    else:
        t = opts.graph_size
        s = opts.begin_size
        N = opts.curriculum_epochs
        graph_size = opts.graph_size

        task_size_linear_ = partial(task_size_linear, s, t, N)
        task_size_logarithmic_ = partial(task_size_logarithmic,s, t, N)

        assert opts.increase_method in ['linear', 'logarithmic']
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            if opts.progressive_distillation:
                # Load small model in each epoch.
                # epoch, size = search_best_model(total_size, radius, opts.save_dir)
                # small_model_path = os.path.join(opts.save_dir, f'epoch-{}-size-{}.pt')
                # small_model = load_small_model_('/epoch-20.pt')
                small_model = load_small_model_('outputs/tsp_20/tsp20_rollout_20210624T190952/epoch-99-size-20.pt')
                run_local_prob_with_fixed_model = partial(run_local_prob, small_model)
                model.run_local_prob = run_local_prob_with_fixed_model
                print(f'beta = {opts.beta}')

            print(f'is curriculum learnin: {opts.curriculum_learning}')
            if opts.curriculum_learning:
                if opts.increase_method == 'linear':
                    graph_size = task_size_linear_(epoch)
                elif opts.increase_method == 'logarithmic':
                    graph_size = task_size_logarithmic_(epoch)
                print(f'epoch: {epoch}, graph size: {graph_size}')
            
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts,
                graph_size
            )

            if opts.progressive_distillation:
                opts.beta *= opts.beta_decay


if __name__ == "__main__":
    run(get_options())
