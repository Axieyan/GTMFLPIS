use pytorch docker image == 21.05
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchtext==0.11.2

fix tensorboard: https://github.com/pytorch/pytorch/issues/22676#issuecomment-586567581
run this python code:

import pkg_resources

for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
    print(entry_point.dist)

[//]: # (pip uninstall nvidia-tensorboard nvidia-tensorboard-plugin-dlprof)
pip install everything that has the word tensorboard
pip install tensorboard

------------------------------
Hardware:
CPU: Intel(R)Core(TM)i7-10710U
Random Access Memory(RAM): 32 Gigabytes
GPU: NVIDIA GeForce RTX 3080 Ti
DisPlay memory: 12 Gigabytes GDR6X Memory
Hary Drive Size:2 Terabytes
Deep Learning Frameworks:Pytorch 1.8.1
Programming Language :Python 3.8
Torch 1.8.1

TorchVison 0.12.0
CUDA 12
Transformers 4.30.2