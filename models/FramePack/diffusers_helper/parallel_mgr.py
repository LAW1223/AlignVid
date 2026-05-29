import torch 
import torch.distributed as dist
import os
SEQ = {}
_SP =None

def split_sequence(input_, dim=1, shape_record="hidden"):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size == 1:
        return input_
    
    tensor_list = torch.chunk(input_, world_size, dim=dim)
    global SEQ
    if not SEQ.get(shape_record, None) and input_.shape[dim] % world_size != 0:
        SEQ[shape_record] = [None] * world_size
        for i in range(world_size):
            SEQ[shape_record][i] = tensor_list[i].shape[dim]
    output = tensor_list[rank].contiguous()
    return output

def gather_sequence(input_, dim=1, shape_record="hidden"):
    input_ = input_.contiguous()
    world_size = dist.get_world_size()
    if world_size == 1:
         return input_
    
    global SEQ
    if not SEQ.get(shape_record, None):
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        b, s, d = input_.shape
        rec_shape = [[b, s, d] for _ in range(world_size)]
        for i, s_i in enumerate(SEQ[shape_record]):
            rec_shape[i][dim]=s_i
        tensor_list = [torch.empty(shape, device=input_.device, dtype=input_.dtype) for shape in rec_shape]
    dist.all_gather(tensor_list, input_)

    output = torch.cat(tensor_list, dim=dim)
    return output

def unset_shape_record(*args):
    global SEQ
    for shape_record in args:
        SEQ[shape_record] = None

def all_to_all(input_,     
    scatter_dim: int,
    shape_record="hidden",
    group=None, 
    ):
    world_size = dist.get_world_size()
    b, s, n, d = input_.shape


    if scatter_dim ==2:
        gather_dim = 1
        n_split = n// world_size
        if not SEQ.get(shape_record, None):
            rec_lst = [torch.empty([b, s, n_split, d], device=input_.device, dtype=input_.dtype) for _ in range(world_size)]
        else:
            rec_lst = [torch.empty([b, SEQ[shape_record][rank_idx], n_split, d], device=input_.device, dtype=input_.dtype) for rank_idx in range(world_size)]
        # send_lst = input_.chunk(world_size, dim=scatter_dim)
        send_lst = [send_t.contiguous() for send_t in torch.chunk(input_, world_size, dim=scatter_dim)]
    
    elif scatter_dim ==1:
        gather_dim =2
        if not SEQ.get(shape_record, None):
            s_split = s // world_size
            rec_lst = [torch.empty([b, s_split, n, d], device=input_.device, dtype=input_.dtype) for _ in range(world_size)]
            send_lst = [send_t.contiguous() for send_t in torch.chunk(input_, world_size, dim=scatter_dim)]
        else:
            rank = dist.get_rank()
            rec_lst = [torch.empty([b, SEQ[shape_record][rank], n, d], device=input_.device, dtype=input_.dtype) for _ in range(world_size)]
            #send_lst = torch.split(input_, SEQ[shape_record], dim=scatter_dim)
            send_lst = [send_t.contiguous() for send_t in torch.split(input_, SEQ[shape_record], dim=scatter_dim)]

    dist.all_to_all(rec_lst, send_lst, group=group)

    out = torch.cat(rec_lst, dim=gather_dim)

    return out

PARALLEL = False
def use_paralle():
    global PARALLEL
    return PARALLEL

def init_parallel_env():
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', 
        world_size=world_size, rank=rank
        )
    global PARALLEL
    PARALLEL = True
    
def init_group(ulys_deg=None):
    global _SP
    _SP =torch.distributed.group.WORLD

def get_ulys_group():
    global _SP
    return _SP
