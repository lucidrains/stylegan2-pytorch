import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from stylegan2_pytorch import Trainer, NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
import wandb as wandb_logger

def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def run_training(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed, log_freq: int = 50):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if model_args.wandb:
        wandb_logger.init()
        wandb_logger.watch(model, log_freq=log_freq)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % log_freq == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()

def train_from_folder(
    data: str = './data',
    results_dir: str = './results',
    models_dir: str = './models',
    name: str = 'default',
    new: bool = False,
    load_from: int = -1,
    image_size: int = 128,
    network_capacity: int = 16,
    fmap_max: int = 512,
    transparent: bool = False,
    batch_size: int = 5,
    gradient_accumulate_every: int = 6,
    num_train_steps: int = 150000,
    learning_rate: float = 2e-4,
    lr_mlp: float = 0.1,
    ttur_mult: float = 1.5,
    rel_disc_loss: bool = False,
    num_workers: int =  None,
    save_every: int = 1000,
    evaluate_every: int = 1000,
    generate: bool = False,
    num_generate: int = 1,
    generate_interpolation: bool = False,
    interpolation_num_steps: int = 100,
    save_frames: bool = False,
    num_image_tiles: int = 8,
    trunc_psi: float = 0.75,
    mixed_prob: float = 0.9,
    fp16: bool = False,
    no_pl_reg: bool = False,
    cl_reg: bool = False,
    fq_layers: list = [],
    fq_dict_size: int = 256,
    attn_layers: list = [],
    no_const: bool = False,
    aug_prob: float = 0.,
    aug_types: list = ['translation', 'cutout'],
    top_k_training: bool = False,
    generator_top_k_gamma: float = 0.99,
    generator_top_k_frac: float = 0.5,
    dual_contrast_loss: bool = False,
    dataset_aug_prob: float = 0.,
    multi_gpus: bool = False,
    calculate_fid_every: int = None,
    calculate_fid_num_images: int = 12800,
    clear_fid_cache: bool = False,
    seed: int = 42,
    log: bool = False,
    wandb: bool = False
):
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dual_contrast_loss = dual_contrast_loss,
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        mixed_prob = mixed_prob,
        log = log,
        wandb = wandb
    )

    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    world_size = torch.cuda.device_count()

    if world_size == 1 or not multi_gpus:
        run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed)
        return

    mp.spawn(run_training,
        args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed),
        nprocs=world_size,
        join=True)

def main():
    fire.Fire(train_from_folder)
