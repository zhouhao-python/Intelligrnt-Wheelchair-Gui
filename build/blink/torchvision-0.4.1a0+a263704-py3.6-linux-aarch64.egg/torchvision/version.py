__version__ = '0.4.1a0+a263704'
git_version = 'a263704079d9d35db8b0966a65ec628d28998bce'
from torchvision import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
