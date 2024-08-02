import traceback
from sklearn.cluster import MiniBatchKMeans
import torch
import argparse
import sys
import os
from pydub import AudioSegment
import subprocess
import numpy as np
import faiss
import logging
logger = logging.getLogger(__name__)

def cpu_count():
    '''Returns the number of CPUs in the system'''
    num = os.cpu_count()
    if num is None:
        raise NotImplementedError('cannot determine number of cpus')
    else:
        return num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extration arguments")

    parser.add_argument("-m", "--model_name", type=str, help="Set model name")
    parser.add_argument("-d", "--dataset_path", type=str, default="./dataset", help="Set dataset folder path")
    parser.add_argument("-sr", "--sampling_rate", type=int, default=40000, help="Set sampling rate")
    parser.add_argument("-f", "--f0_method", type=str, default="rmvpe_gpu", help="Set f0 method")
    parser.add_argument("-nc", "--n_cpu_process", type=int, default=np.ceil(cpu_count() / 1.5), help="Set number of CPU processes for pitch processing and extraction")
    parser.add_argument("-half", "--half", type=bool, default=True, help="Set half precision")
    parser.add_argument("-rv", "--rvc_version", type=str, default="v2", help="Set RVC version of the model")
    args = parser.parse_args()

    if args.model_name is None:
        raise Exception("Model name is required. Use -m or --model_name to set model name.")

    model_name = args.model_name
    dataset_path = os.path.abspath(args.dataset_path) if args.dataset_path == "./dataset" else args.dataset_path
    sampling_rate = args.sampling_rate
    f0_method = args.f0_method
    n_cpu_process = int(args.n_cpu_process)
    half_precision = args.half
    rvc_version = args.rvc_version

    gpus = "-"
    gpus_rmvpe = "-"

    if not torch.cuda.is_available():
        raise Exception("CUDA is not available. Please check your GPU availability.")
    else:
        for i in range(torch.cuda.device_count()):
            print("Feature Extraction GPU")
            print("Available GPU(s):")
            print(f"{(i)} - {torch.cuda.get_device_name(i)}\n")
            gpus = input("Enter the GPU index(es) separated by \"-\", e.g., 0-1-2 to use GPU 0, 1, and 2: ")

            gpu_indices = [int(gpu) for gpu in gpus.split("-")]

            # Check if all indices are valid
            if not all([gpu in range(torch.cuda.device_count()) for gpu in gpu_indices]):
                raise ValueError("One or more specified GPU indices are out of range.")
            
            # Check for duplicates let say 0-0 or 1-1
            for i in range(len(gpu_indices)):
                if gpu_indices.count(i) > 1:
                    raise ValueError("Duplicate GPU indices are not allowed.")
            
    if f0_method == "rmvpe_gpu":
        for i in range(torch.cuda.device_count()):
            print("\nRMVPE GPU")
            print("Available GPU(s):")
            print(f"{(i)} - {torch.cuda.get_device_name(i)}\n")
            print("GPU index(es) separated by \"-\", e.g., 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1. Input 0 for single GPU.")
            
            gpus_rmvpe = input("Enter the GPU index(es): ")

            gpu_indices = [int(gpu) for gpu in gpus_rmvpe.split("-")]

            # Check if all indices are valid
            if not all([gpu in range(torch.cuda.device_count()) for gpu in gpu_indices]):
                raise ValueError("One or more specified RMVPE GPU indices are out of range.")

    if sampling_rate not in [40000, 48000]:
        raise Exception("Sample rate must be one of 40000 or 48000.")

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
    log_path = os.path.abspath(log_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # A boolean value for parallel processing.
    no_parallel = False 
    # A floating-point value for preprocessing. (e.g., percentage)
    preprocess_percentage = 3.0

    command = [sys.executable, "infer/modules/train/preprocess.py", dataset_path, str(sampling_rate), str(n_cpu_process), log_path, str(no_parallel), str(preprocess_percentage)]

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()

    print(stdout)
    print(stderr)

    p.wait()
        # f.write("\nPreprocessing finished")

    with open(f"./logs/{model_name}/preprocess.log", "r") as f:
        if "end preprocess" in f.read():
            print("✅ Preprocessing successful")
        else:
            print("❌ Preprocessing failed")

    # Extract F0 and features
    if f0_method == "rmvpe_gpu":
        gpus_rmvpe = gpus_rmvpe.split("-")
        
        for idx, gpu in enumerate(gpus_rmvpe):
            command = [sys.executable, "infer/modules/train/extract/extract_f0_rmvpe.py", str(len(gpus_rmvpe)), str(idx), str(gpu), log_path, str(half_precision)]
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            stdout, stderr = p.communicate()

            print(stdout)
            print(stderr)

            p.wait()
    else:
        command = [sys.executable, "infer/modules/train/extract/extract_f0_print.py", log_path, str(n_cpu_process), f0_method]
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()

        # print(stdout)
        # print(stderr)

        p.wait()
        

    cuda = "cuda:0"

    for idx, gpu in enumerate(gpus):
        command = [sys.executable, "infer/modules/train/extract_feature_print.py", str(cuda), str(len(gpus)), str(idx), str(gpu), log_path, str(rvc_version), str(half_precision)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()

        # print(stdout)
        # print(stderr)

        p.wait()

    with open(f"./logs/{model_name}/extract_f0_feature.log", "r") as f:
        if "all-feature-done" in f.read():
            print("✅ Extracting f0 feature successful")
        else:
            print("❌ Extracting f0 feature failed")

    def train_index(experiment_dir, version):
        exp_dir = "logs/%s" % (experiment_dir)
        os.makedirs(exp_dir, exist_ok=True)
        feature_dir = (
            "%s/3_feature256" % (exp_dir)
            if version == "v1"
            else "%s/3_feature768" % (exp_dir)
        )
        if not os.path.exists(feature_dir):
            return "请先进行特征提取!"
        listdir_res = list(os.listdir(feature_dir))
        if len(listdir_res) == 0:
            return "请先进行特征提取！"
        infos = []
        npys = []
        for name in sorted(listdir_res):
            phone = np.load("%s/%s" % (feature_dir, name))
            npys.append(phone)
        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]
        if big_npy.shape[0] > 2e5:
            infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
            yield "\n".join(infos)
            try:
                big_npy = (
                    MiniBatchKMeans(
                        n_clusters=10000,
                        verbose=True,
                        batch_size=256 * n_cpu_process,
                        compute_labels=False,
                        init="random",
                    )
                    .fit(big_npy)
                    .cluster_centers_
                )
            except:
                info = traceback.format_exc()
                logger.info(info)
                infos.append(info)
                yield "\n".join(infos)

        np.save("%s/total_fea.npy" % exp_dir, big_npy)
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        infos.append("%s,%s" % (big_npy.shape, n_ivf))
        yield "\n".join(infos)
        index = faiss.index_factory(256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf)
        infos.append("training")
        yield "\n".join(infos)
        index_ivf = faiss.extract_index_ivf(index)  #
        index_ivf.nprobe = 1
        index.train(big_npy)
        faiss.write_index(
            index,
            "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (exp_dir, n_ivf, index_ivf.nprobe, experiment_dir, version),
        )

        infos.append("adding")
        yield "\n".join(infos)
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])
        faiss.write_index(
            index,
            "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (exp_dir, n_ivf, index_ivf.nprobe, experiment_dir, version),
        )
        infos.append(
            "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (n_ivf, index_ivf.nprobe, experiment_dir, version)
        )
    
    training_log = train_index(model_name, rvc_version)

    for line in training_log:
        print(line)
        if 'adding' in line:
            print("✅ Training index successful")
