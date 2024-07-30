import torch
import argparse
import os
from pydub import AudioSegment
import subprocess
from datetime import datetime

import numpy as np
import faiss

parser = argparse.ArgumentParser(description="Training arguments")

parser.add_argument("-m", "--model_name", type=str, help="Set model name")
parser.add_argument("-d", "--dataset_path", type=str, default="./dataset", help="Set dataset folder path")
parser.add_argument("-sr", "--sampling_rate", type=str, default="40000", help="Set sampling rate")
parser.add_argument("-f", "--f0_method", type=str, default="rmvpe_gpu", help="Set f0 method")
parser.add_argument("-c", "--cpu", type=int, default=8, help="Set number of CPU processes for pitch processing and extraction")
parser.add_argument("-half", "--half", type=bool, default=True, help="Set half precision")
parser.add_argument("-rv", "--rvc_version", type=str, default="v2", help="Set RVC version of the model")
args = parser.parse_args()

model_name = args.model_name
dataset_path = os.path.abspath(args.dataset_folder)
sampling_rate = args.sampling_rate
f0_method = args.f0_method
n_cpu_process = args.cpu
half_precision = args.half
rvc_version = args.rvc_version

gpus = "-"
gpus_rmvpe = "-"

if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Please check your GPU availability.")
else:
    for i in range(torch.cuda.device_count()):
        print(f"{i} {torch.cuda.get_device_name(i)}")

        gpus = input("Enter the GPU index(es) separated by \"-\", e.g., 0-1-2 to use GPU 0, 1, and 2: ")

        gpu_indices = [int(gpu) for gpu in gpus.split("-")]

        # Check if all indices are valid
        if not all([gpu in range(torch.cuda.device_count()) for gpu in gpu_indices]):
            raise ValueError("One or more specified GPU indices are out of range.")
        
        # Check for duplicates
        if len(gpu_indices.split("-")) != len(set(gpu_indices)):
            raise ValueError("Duplicate GPU indices are not allowed.")

if f0_method == "rmvpe_gpu":
    for i in range(torch.cuda.device_count()):
        print(f"{i} {torch.cuda.get_device_name(i)}")
        print("\nEnter the GPU index(es) separated by \"-\", e.g., 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1. Input 0 for single GPU.")
        
        gpus_rmvpe = input("Enter the GPU index(es): ")

        gpu_indices = [int(gpu) for gpu in gpus_rmvpe.split("-")]

        # Check if all indices are valid
        if not all([gpu in range(torch.cuda.device_count()) for gpu in gpu_indices]):
            raise ValueError("One or more specified RMVPE GPU indices are out of range.")

if sampling_rate not in ["40000", "48000"]:
    raise Exception("Sample rate must be one of 40000, 44100, 48000 or 96000.")

if f0_method not in ["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"]:
    raise Exception("F0 method must be one of pm, harvest, dio, rmvpe or rmvpe_gpu.")

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

def get_audio_duration(ds_path):
    duration = 0
    for root, dirs, files in os.walk(ds_path):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3") or file.endswith(".flac"):
                audio_path = os.path.join(root, file)
                audio = AudioSegment.from_file(audio_path)
                duration += audio.duration_seconds
    return duration

try:
    duration = get_audio_duration(dataset_path)
    if duration < 600:
        cache = False
    else:
        cache = True
except:
    cache = False

if (len(os.listdir(dataset_path)) < 1):
    raise Exception("No audio files found in dataset folder")

log_path = f"./logs/{model_name}"

if not os.path.exists(log_path):
    os.makedirs(log_path)

formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f"./logs/{model_name}/preprocess_{formatted_time}.log","w") as f:
    f.write("Preprocessing started")

    # A boolean value for parallel processing.
    no_parallel = False 
    # A floating-point value for preprocessing. (e.g., percentage)
    preprocess_percentage = 3.0

    command = ["python", f"infer/modules/train/preprocess.py \"{dataset_path}\" {sampling_rate} {n_cpu_process} \"{log_path}\" {no_parallel} {preprocess_percentage}"]

    p = subprocess.Popen(command, stdout=f, stderr=f)
    p.wait()

    f.write("Preprocessing finished")

with open(f"./logs/{model_name}/preprocess_{formatted_time}.log", "r") as f:
    if "end preprocess" in f.read():
        print("✅ Preprocessing successful")
    else:
        print("❌ Preprocessing failed")

with open(f"./logs/{model_name}/extract_f0_feature_{formatted_time}.log", "w") as f:
    f.write("Extracting F0 feature started")

    if f0_method == "rmvpe_gpu":
        gpus_rmvpe = gpus_rmvpe.split("-")
        # is_half_precision = True
        
        for idx, gpu in enumerate(gpus_rmvpe):
            command = ["python", f"infer/modules/train/extract_f0_rmvpe.py {len(gpus_rmvpe)} {idx} {gpu} \"{log_path}\" {half_precision}"]
            subprocess.Popen(command, stdout=f, stderr=f).wait()
            
    else:
        command = ["python", f"infer/modules/train/extract_f0_print.py \"{log_path}\" {n_cpu_process} {f0_method}"]
        subprocess.Popen(command, stdout=f, stderr=f).wait()
        
    cuda = "cuda:0"
    
    for idx, gpu in enumerate(gpus):
        command = ["python", f"infer/modules/train/extract_feature_print.py {cuda} {len(gpus)} {idx} {gpu} \"{log_path}\" {rvc_version} {half_precision}"]
        subprocess.Popen(command, stdout=f, stderr=f).wait()

    f.write("Extracting F0 feature finished")

with open(f"./logs/{model_name}/extract_f0_feature_{formatted_time}.log", "r") as f:
    if "all-feature-done" in f.read():
        print("✅ Extracting f0 feature successful")
    else:
        print("❌ Extracting f0 feature failed")



