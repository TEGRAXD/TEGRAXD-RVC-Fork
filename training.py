import torch
import argparse
import os
import subprocess
import sys
import json
import pathlib
from random import shuffle

parser = argparse.ArgumentParser(description="Training arguments")

parser.add_argument("-m", "--model_name", type=str, help="Set model name")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Set epochs")
parser.add_argument("-s", "--save_freq", type=int, default=10, help="Set save frequency")
parser.add_argument("-b", "--batch_size", type=int, default=5, help="Set batch size per GPU")
parser.add_argument("-sr", "--sample_rate", type=str, default="40k", help="Set sample rate")
parser.add_argument("-c", "--cache_training_sets", type=bool, help="Set cache all training sets to GPU memory. (Dataset less than 10 mins is recommended to set True)")
parser.add_argument("-rv", "--rvc_version", type=str, default="v2", help="Set RVC version of the model")
args = parser.parse_args()

if args.model_name is None:
    raise Exception("Model name is required. Use -m or --model_name to set model name.")

if args.cache_training_sets is None:
    raise Exception("Cache path is required. Use -c or --cache_training_sets to set cache training sets to GPU.")

if args.sample_rate not in ["40k", "48k"]:
    raise Exception("Sample rate must be 40k or 48k.")

now_dir = os.getcwd()

model_name = args.model_name
epochs = args.epochs
save_freq = args.save_freq
batch_size = args.batch_size
sample_rate = args.sample_rate
cache = args.cache_training_sets
rvc_version = args.rvc_version

gpus = "-"

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

G_file = f"assets/pretrained_v2/f0G{sample_rate}.pth"
D_file = f"assets/pretrained_v2/f0D{sample_rate}.pth"

if (not os.path.exists(G_file)) or (not os.path.exists(D_file)):
    print("❌ Pretrained model not found")
    exit(0)

def train(
    experiment_dir,
    sr,
    use_f0,
    speaker_id,
    save_epoch,
    epochs,
    batch_size,
    save_latest,
    pretrained_G,
    pretrained_D,
    gpus,
    cache,
    save_every_weight,
    version,
):
    exp_dir = "%s/logs/%s" % (now_dir, experiment_dir)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if use_f0:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if use_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    speaker_id,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    speaker_id,
                )
            )
    fea_dim = 256 if version == "v1" else 768
    if use_f0:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr, now_dir, fea_dim, now_dir, now_dir, speaker_id)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr, now_dir, fea_dim, speaker_id)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))

    # Replace logger.debug, logger.info with print statements
    print("Write filelist done")
    print("Use gpus:", str(gpus))
    if pretrained_G == "":
        print("No pretrained Generator")
    if pretrained_D == "":
        print("No pretrained Discriminator")
    if version == "v1" or sr == "40k":
        config_path = "configs/v1/%s.json" % sr
    else:
        config_path = "configs/v2/%s.json" % sr
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                json.dump(
                    config_data,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
            f.write("\n")

    cmd = (
        '%s infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
        % (
            sys.executable,
            experiment_dir,
            sr,
            1 if use_f0 else 0,
            batch_size,
            gpus,
            epochs,
            save_epoch,
            "-pg %s" % pretrained_G if pretrained_G != "" else "",
            "-pd %s" % pretrained_D if pretrained_D != "" else "",
            1 if save_latest == True else 0,
            1 if cache == True else 0,
            1 if save_every_weight == True else 0,
            version,
        )
    )
    # Use PIPE to capture the output and error streams
    p = subprocess.Popen(cmd, shell=True, cwd=now_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

    # Print the command's output as it runs
    for line in p.stdout:
        print(line.strip())

    # Wait for the command to complete
    p.wait()
    return 'At the end of the training, you can view the training log on the console or the "train.log" in the experiment folder.'

training_log = train(
    model_name,
    sample_rate,
    True,
    0,
    save_freq,
    epochs,
    batch_size,
    True,
    G_file,
    D_file,
    gpus,
    cache,
    True,
    rvc_version,
)

print(training_log)
