# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from omegaconf import OmegaConf as om

from examples.llm.src import COMPOSER_MODEL_REGISTRY

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    te_installed = True
except ImportError:
    te_installed = False


def build_composer_model(model_cfg, tokenizer_cfg, tensor_parallel_group=None):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    try:
        return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer_cfg, tensor_parallel_group)
    except:
        raise ValueError(
            f'Not sure how to build model with name={model_cfg.name}')


def test_throughput(rank, world_size):
    torch.cuda.set_device(rank)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        f'torch.distributed.*_base is a private function and will be deprecated.*'
    )

    # get configs
    yaml_path, args_list = sys.argv[2], sys.argv[3:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    
    # updt config
    cfg.model.init_device = 'cpu'
    cfg.pop('fsdp_config')
    cfg.model.te_tx_layer = False
    cfg.model.te_linears = False

    fp8 = False
    if fp8 and not (cfg.model.te_tx_layer or cfg.model.te_linears):
        warnings.warn('TE layers not being used; nothing will cast to fp8.')

    itrs = cfg.get('itrs', 20)
    time_start_itr = cfg.get('time_start_itr', 4)
    dtype = torch.bfloat16 if 'bf' in cfg.precision else torch.float16
    device_type = 'a100' if 'a100' in torch.cuda.get_device_name(0).lower() else 'h100'

    print(f'{rank=}, {world_size=}')

    if (cfg.model.te_tx_layer or cfg.model.te_linears) and not te_installed:
        raise ValueError(
            "Transformer Engine is not installed! "
            "Please install transformer engine with "
            "pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@stable")

    world_group = dist.init_process_group('nccl', rank=rank, world_size=world_size)
    data_parallel_group = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")
    tensor_parallel_group = torch.distributed.new_group(ranks=[rank], backend="nccl")

    # Build Model
    if rank == 0: print('Initializing model...')
    model = build_composer_model(cfg.model, cfg.tokenizer, tensor_parallel_group)
    cfg.n_params = sum(p.numel() for p in model.parameters())
    if rank == 0: print(f'{cfg.n_params=:.2e}')

    model.to(torch.device(f'cuda:{rank}'))

    # construct DDP model
    if rank == 0: print(f'setting up DDP model')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, process_group=data_parallel_group)
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # define optimizer
    if rank == 0: print(f'init optimizer')
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.to(dtype=dtype)

    if rank == 0: print(model)

    batch = {}
    batch['input_ids'] = torch.randint(low=0, high=cfg.model.vocab_size, size=(cfg.device_train_microbatch_size, cfg.max_seq_len)).to(rank)
    batch['labels'] = batch['input_ids'].clone().detach().to(rank)
    batch['attention_mask'] = torch.ones(cfg.device_train_microbatch_size, cfg.max_seq_len, dtype=torch.bool).to(rank)

    if not fp8:
        def fwd_loss(batch):
            with torch.autocast('cuda', dtype=dtype, enabled=True):
                outputs = model(batch)
                loss = module.loss(outputs, batch)
            return loss
    else:
        def fwd_loss(batch):
            with torch.autocast('cuda', dtype=dtype, enabled=True):
                with te.fp8_autocast(
                    enabled=True,
                    fp8_recipe=DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"),
                    fp8_group=data_parallel_group,
                ):
                    outputs = model(batch)
                    loss = module.loss(outputs, batch)
            return loss

    torch.cuda.current_stream().synchronize()
    for itr in range(itrs):
        if itr == time_start_itr:
            t0 = time.time()

        # forward pass
        loss = fwd_loss(batch)
        if rank == 0: print(f'{loss.item()=}')

        # backward pass
        loss.backward()
        # update parameters
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.current_stream().synchronize()
    elapsed_time = time.time() - t0

    throughput = cfg.device_train_microbatch_size * cfg.max_seq_len * (itrs - time_start_itr) / elapsed_time
    if rank == 0: print(f'{throughput=:.4f} tok/sec')

    fpb = module.flops_per_batch(batch)
    flopsps = fpb * (itrs - time_start_itr) / elapsed_time
    if rank == 0: print(f'{flopsps/1e12:.4f} TFLOPs/sec')

    if device_type == 'a100':
        if rank == 0: print(f'MFU: {100 * flopsps / 312e12:.4f}%')
    else:
        if rank == 0: print(f'MFU (fp16): {100 * flopsps / 1e15:.4f}%')
        if fp8 and rank == 0: print(f'MFU (fp8): {100 * flopsps / 2e15:.4f}%')


if __name__=="__main__":
    # Environment variables which need to be set when using c10d's default "env" initialization mode.
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29500"

    world_size = int(sys.argv[1])
    mp.spawn(test_throughput,
        args=(world_size,),
        nprocs=world_size,
        join=True)
