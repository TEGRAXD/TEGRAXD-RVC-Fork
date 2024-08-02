import argparse

parser = argparse.ArgumentParser(description="RVC Inference arguments")

parser.add_argument("-t", "--target_path", type=str, help="Set target audio file path with extension")
parser.add_argument("-m", "--model_name", type=str, help="Set model name")
args = parser.parse_args()

if args.target_path is None:
    raise Exception("Target path is required. Use -t or --target_path to set target audio file path with extension.")

if args.model_name is None:
    raise Exception("Model name is required. Use -m or --model_name to set model name.")

target_path = args.target_path
model_name = args.model_name
model_filename = f"{model_name}.pth"