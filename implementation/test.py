import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
import os
print(os.environ['NVIDIA_VISIBLE_DEVICES'])