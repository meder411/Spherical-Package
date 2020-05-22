import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def dprint(*args, **kwargs):
    # Prints only from process 0
    if get_rank() == 0:
        print(*args, **kwargs)


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def initialize(local_rank):
    dprint('Initializing distributed group')
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
