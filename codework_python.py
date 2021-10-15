# -*- coding: utf-8 -*-
"""
@author: Hakan Hekimgil

Codes I commonly use in Python
"""

# libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from tqdm import tqdm
# from datetime import datetime, timedelta
# from random import sample
import random
# import copy
# import pickle

# import torch
# import torch.nn as nn
# # import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils import data
# from torchvision import transforms
# import torchvision.models as models
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

# environment info
print("Environment Info:")
print("-----------------")
print("Python version:      ",
      f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}-{sys.version_info[3]}")
print("Numpy Version:       ", np.__version__)
print("Pandas Version:      ", pd.__version__)
# print("OpenCV Version:      ", cv2.__version__)
# print("Tesseract Version:   ", pytesseract.get_tesseract_version().vstring.split("\r\n ")[0])
# print("PyTesseract Version: ", pytesseract.__version__)
# print("Seaborn Version:     ", sns.__version__)
# print("PyTorch Version: ", torch.__version__)
print()

# DO NOT FORGET TO SET THE CONSOLE WORKING DIRECTORY IN SPYDER!!!
assert os.getcwd()[-8:] == "Codework"

# folder locations
basfolder = os.getcwd()
datfolder = os.path.join(basfolder, "data", "")

outfolder = os.path.join(basfolder, "outputs", "")
repfolder = os.path.join(basfolder, "reports", "")
pklfolder = os.path.join(basfolder, "pickles", "")
modfolder = os.path.join(basfolder, "models", "")
profolder = os.path.join(basfolder, "process", "")
cptfolder = os.path.join(profolder, "checkpoints", "")


# for coloring, see:
# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
# https://en.wikipedia.org/wiki/ANSI_escape_code

CGRN = "\033[1;32m"
CRED = "\033[1;31m"
CYEL = "\033[1;33m"
CCYN = "\033[1;36m"
CEND = "\033[0m"

CMGN = "\033[1;35m"
CBLU = "\033[1;34m"
CGRNBND = "\033[1;42m"
CBLUBND = "\033[1;44m"
CMGNBND = "\033[1;45m"

# print(CMGN + "TEST COLOR" + CEND)


# settings
verbose = True
debug = False
timing = False


# # set up Cuda
# cuda_available = torch.cuda.is_available()
# print("CUDA available?:  ", cuda_available)
# if cuda_available and debugging_mode:
#     print("----------------------------------------------------------------")
#     print("***** CUDA available but switching to CPU for debugging... *****")
#     print("----------------------------------------------------------------|n")
#     cuda_available = False
# if cuda_available:
#     device = torch.device("cuda")
#     cudaname = torch.cuda.get_device_name(0)
#     print("Cuda device name: ", cudaname)
#     print("Cuda capability:  ", torch.cuda.get_device_capability())
#     print("Memory Usage      ")
#     print("Allocated:        ", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
#     print("Reserved:         ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
#     # print("Cached:           ", round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
#     # print("Summary:          ", torch.cuda.memory_summary())
#     # torch.cuda.ipc_collect()
#     # torch.cuda.empty_cache()
# else:
#     device = torch.device("cpu")
#
# # cuda names: "GeForce GTX 1060", "GeForce GTX 950M"
# if cudaname == "NVIDIA GeForce GTX 1060":
#     machine = "asus_rog"
#     mac_des = "rog"
# elif cudaname == "NVIDIA GeForce GTX 950M":
#     machine = "asus_24"
#     mac_des = "a24"
# elif cudaname == "NVIDIA GeForce RTX 2070 Super":
#     machine = "aw_m17"
#     mac_des = "m17"
# else:
#     machine = "undefined"
#     mac_des = "unk"
#     print(f"missing definition for machine with {cudaname}")
#
# # use batchsizes appropriate for the GPU memory size
# # taking the image sizes into consideration
# if not cuda_available:
#     batchsize = 64
# else:
#     if machine == "asus_24":
#         batchsize = 64
#     elif machine == "asus_rog":
#         batchsize = 128
#     elif machine == "aw_m17":
#         batchsize = 64
#     else:
#         print(f"missing definition for machine with {cudaname}")
#         batchsize = 64


"""
Basic python stuff
"""

test_list = random.sample(range(1, 100), 10)
print(sorted(test_list))
print(test_list)
test_list.sort()
print(test_list)

for idx, item in enumerate(test_list):
    print(f"{idx:2,d}: {item:2,d}")


"""
Plotting
"""


plt.plot(test_list)
plt.title('Sample plot')
plt.ylabel('y labels')
plt.xlabel('x labels')
plt.show()

