import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object

print("what?")

dist.init_process_group(backend='nccl')

accelerator = Accelerator()

# each GPU creates a string
message=[ f"Hello this is GPU {accelerator.process_index}" ] 

# collect the messages from all GPUs
messages=gather_object(message)

# output the messages only on the main process with accelerator.print() 
accelerator.print(messages)

dist.destroy_process_group()