# %%
import transformers
import accelerate
import peft
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-lr', "--lr", help="learning rate", type=float, default='0.005')
parser.add_argument('-cp', "--checkpoint", help="check point", type=bool)
args = parser.parse_args()

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
"Transformers version: 4.27.4"
"Accelerate version: 0.18.0"
"PEFT version: 0.2.0"

# %%
model_checkpoint = "google/vit-base-patch16-224-in21k"

# %%

# from datasets import load_dataset

# # dataset = load_dataset("food101", split="train[:5000]")


# #数据集下载到硬盘disk
# from datasets import load_dataset
# dataset = load_dataset("food101", cache_dir="./datasets")
# dataset.save_to_disk('datasets/food101')



# %%
#加载数据集.
import datasets

dataset = datasets.load_from_disk("datasets/food101")
#print(f'{type(dataset)=}')
print(dataset.keys())

# %%
labels = dataset['train'].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]
#"baklava"

# %%
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# %%
#splits = dataset.train_test_split(test_size=0.1)
# train_ds = splits["train"]
# val_ds = splits["test"]
train_ds = dataset['train']
val_ds = dataset['validation']
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# %%
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# %%

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
print_trainable_parameters(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)


# %%
print(model)

# %%
lora_model = get_peft_model(model,config)
print_trainable_parameters(model)

# %%
from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]
batch_size = 128

if not os.path.exists('./checkpoint_'+ args.lr):
    os.mkdir('./checkpoint_'+ args.lr)
train_args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    output_dir = './checkpoint_'+ args.lr,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
)

# %%
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# %%
#Define collation function
#A collation function is used by Trainer to gather a batch of training and evaluation examples and prepare them in a format that is acceptable by the underlying model.

import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}




# %%
trainer = Trainer(
    lora_model,
    train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train(resume_from_checkpoint=args.checkpoint)


trainer.evaluate(val_ds)

if not os.path.exists("./pretraining_model"):
    torch.save(lora_model.state_dict,f"./pretraining_model/{args.lr}_huggingface_lora_model.pth")
else:
    os.mkdir("./pretraining_model")
    torch.save(lora_model.state_dict,f"./pretraining_model/{args.lr}_huggingface_lora_model.pth")



