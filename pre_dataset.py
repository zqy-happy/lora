from datasets import load_dataset
dataset = load_dataset("food101", cache_dir="./datasets")
dataset.save_to_disk('datasets/food101')