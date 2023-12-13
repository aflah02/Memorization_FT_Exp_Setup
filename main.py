import argparse
import numpy as np
import os
import itertools
import pandas as pd
import random
import math
import wandb
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    set_seed,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    AutoModelForCausalLM, 

)
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers.utils import ModelOutput
from datasets import Dataset, DatasetDict, load_dataset
from peft import (get_peft_config, get_peft_model, LoraConfig, TaskType, IA3Config, 
                  PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, PromptTuningInit,
                  IA3Model)
import torch
import torch.nn.functional as F
from custom_datasets import *

WANDB_PROJECT_NAME = "PEFT-Memorization-Analysis"

# Choose fine-tuning task from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--fine_tuning_method", type=str, default="full-finetuning", help="Choose one of the fine-tuning methods: full-finetuning, prefix-tuning, prompt-tuning, lora, ia3, p-tuning or freeze-subset")
parser.add_argument("--model", type=str, default="pythia-1b", help="Provide the model id (Choose pythia model)")
parser.add_argument("--task", type=str, default="random_strings", help="Provide the task name")
parser.add_argument("--seq_len", type=int, default=128, help="Provide the sequence length")
parser.add_argument("--num_seq", type=int, default=8, help="Provide the number of training examples")
parser.add_argument("--num_train_epochs", type=int, default=1, help="Provide the number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Provide the batch size")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Provide the learning rate")
parser.add_argument("--bias_trainable", type=bool, default=True, help="Provide the bias_trainable")
parser.add_argument("--layer_norm_trainable", type=bool, default=True, help="Provide the layer_norm_trainable")
parser.add_argument("--mlp_trainable", type=bool, default=True, help="Provide the mlp_trainable")
parser.add_argument("--attention_trainable", type=bool, default=True, help="Provide the attention_trainable")
parser.add_argument("--layers_trainable", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", help="Provide the layers to train")
parser.add_argument("--embed_in_trainable", type=bool, default=True, help="Provide the embed_in_trainable")
parser.add_argument("--embed_out_trainable", type=bool, default=True, help="Provide the embed_out_trainable")
parser.add_argument("--seed", type=int, default=42, help="Provide the seed")
parser.add_argument("--num_virtual_tokens", type=int, default=8, help="Provide the number of virtual tokens")
parser.add_argument("--encoder_hidden_size_p_tuning", type=int, default=128, help="Provide the encoder_hidden_size for p-tuning")
parser.add_argument("--init_text_prompt_tuning", type=str, default="This is a test", help="Provide the init_text for prompt tuning")
parser.add_argument("--target_modules_IA3", type=str, default="k_proj,v_proj,down_proj", help="Provide the target_modules for IA3")
parser.add_argument("--feed_forward_modules_IA3", type=str, default="down_proj", help="Provide the feed_forward_modules for IA3")
parser.add_argument("--r_lora", type=int, default=8, help="Provide the r value for lora")
parser.add_argument("--alpha_lora", type=int, default=32, help="Provide the alpha value for lora")
parser.add_argument("--dropout_lora", type=float, default=0.1, help="Provide the dropout value for lora")
parser.add_argument("--gpu", type=int, default=0, help="Provide the gpu id")

# Get the arguments
args = parser.parse_args()
fine_tuning_method = args.fine_tuning_method
model_id = args.model
task_name = args.task
seq_len = args.seq_len
num_seq = args.num_seq
num_train_epochs = args.num_train_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
bias_trainable = args.bias_trainable
layer_norm_trainable = args.layer_norm_trainable
mlp_trainable = args.mlp_trainable
attention_trainable = args.attention_trainable
layers_trainable = args.layers_trainable
embed_in_trainable = args.embed_in_trainable
embed_out_trainable = args.embed_out_trainable
seed = args.seed
num_virtual_tokens = args.num_virtual_tokens
encoder_hidden_size_p_tuning = args.encoder_hidden_size_p_tuning
init_text_prompt_tuning = args.init_text_prompt_tuning
target_modules_IA3 = args.target_modules_IA3
feed_forward_modules_IA3 = args.feed_forward_modules_IA3
r_lora = args.r_lora
alpha_lora = args.alpha_lora
dropout_lora = args.dropout_lora
gpu = args.gpu

layers_trainable = [int(layer) for layer in layers_trainable.split(",")]
target_modules_IA3 = target_modules_IA3.split(",")
feed_forward_modules_IA3 = feed_forward_modules_IA3.split(",")

# Error Checks
if model_id != "pythia-1b":
    raise ValueError("Invalid model id. Please provide a valid model id")

if fine_tuning_method not in ["full-finetuning", "prefix-tuning", "prompt-tuning", "lora", "ia3", "p-tuning", "freeze-subset"]:
    raise ValueError("Invalid fine-tuning method")

if fine_tuning_method == "freeze-subset":
    if bias_trainable and layer_norm_trainable and mlp_trainable and attention_trainable and embed_in_trainable and embed_out_trainable and len(layers_trainable) == 16:
        raise ValueError("All the layers are trainable. Please set at least one layer to be frozen. If you want to train all the layers, use full-finetuning method")
    
# Set the seed
set_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_id}")
tokenizer.pad_token = tokenizer.eos_token

params_dict = {
    "task": task_name,
    "model": model_id,
    "fine_tuning_method": fine_tuning_method,
    "seq_len": seq_len,
    "num_seq": num_seq,
    "num_train_epochs": num_train_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "bias_trainable": bias_trainable,
    "layer_norm_trainable": layer_norm_trainable,
    "mlp_trainable": mlp_trainable,
    "attention_trainable": attention_trainable,
    "layers_trainable": layers_trainable,
    "embed_in_trainable": embed_in_trainable,
    "embed_out_trainable": embed_out_trainable,
    "seed": seed,
    "num_virtual_tokens": num_virtual_tokens,
    "encoder_hidden_size_p_tuning": encoder_hidden_size_p_tuning,
    "init_text_prompt_tuning": init_text_prompt_tuning,
    "target_modules_IA3": target_modules_IA3,
    "feed_forward_modules_IA3": feed_forward_modules_IA3,
    "r_lora": r_lora,
    "alpha_lora": alpha_lora,
    "dropout_lora": dropout_lora,
    "gpu": gpu,
}

run_name = "Task: " + task_name + " Model: " + model_id + " Fine-tuning method: " + fine_tuning_method + " Sequence length: " + str(seq_len)\
         + " Number of sequences: "+ str(num_seq) + " Number of epochs: " + str(num_train_epochs) + " Batch size: " + str(batch_size)\
         + " Learning rate: " + str(learning_rate) + " Seed: " + str(seed)
if fine_tuning_method == "prefix-tuning":
    run_name += " Num virtual tokens: " + str(num_virtual_tokens)
elif fine_tuning_method == "prompt-tuning":
    run_name += " Num virtual tokens: " + str(num_virtual_tokens) + " Init text: " + init_text_prompt_tuning
elif fine_tuning_method == "ia3":
    run_name += " Target modules: " + target_modules_IA3 + " Feed forward modules: " + feed_forward_modules_IA3
elif fine_tuning_method == "lora":
    run_name += " r: " + str(r_lora) + " alpha: " + str(alpha_lora) + " dropout: " + str(dropout_lora)
elif fine_tuning_method == "p-tuning":
    run_name += " Encoder hidden size: " + str(encoder_hidden_size_p_tuning) + " Num virtual tokens: " + str(num_virtual_tokens)
elif fine_tuning_method == "freeze-subset":
    run_name += " Bias trainable: " + str(bias_trainable) + " Layer norm trainable: " + str(layer_norm_trainable) + " MLP trainable: " + str(mlp_trainable)\
                + " Attention trainable: " + str(attention_trainable) + " Layers trainable: " + str(layers_trainable) + " Embed in trainable: " + str(embed_in_trainable)\
                + " Embed out trainable: " + str(embed_out_trainable)
elif fine_tuning_method == "full-finetuning":
    run_name += " Full finetuning"

os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
os.environ["WANDB_WATCH"] = "all"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.init(project=WANDB_PROJECT_NAME)
wandb.run.name = run_name
wandb.config.update(params_dict)

if task_name == "random_strings":
    dataset = generate_random_string_dataset(seed=seed, num_sequences=num_seq, sequence_length=seq_len)
else:
    raise ValueError("Invalid task name")

# Encode the dataset
encoded_dataset = encode_character_wise(tokenizer, dataset)

model_id = 'EleutherAI/' + model_id

# Get the model
model = AutoModelForCausalLM.from_pretrained(model_id)

# Get the config
if fine_tuning_method == "IA3":
    config = IA3Config(peft_type="IA3", task_type=TaskType.CAUSAL_LM, target_modules=target_modules_IA3, feed_forward_modules=feed_forward_modules_IA3)
    model = IA3Model(model, config)
elif fine_tuning_method == "lora":
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r_lora, alpha=alpha_lora, dropout=dropout_lora, inference_mode=False)
    model = get_peft_model(model, config)
elif fine_tuning_method == "prefix-tuning":
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens, inference_mode=False)
    model = get_peft_model(model, config)
elif fine_tuning_method == "prompt-tuning":
    config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens, prompt_tuning_init=PromptTuningInit.TEXT, init_text=init_text_prompt_tuning, tokenizer_name_or_path=model_id)
    model = get_peft_model(model, config)
elif fine_tuning_method == "p-tuning":
    config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens, encoder_hidden_size=encoder_hidden_size_p_tuning)
    model = get_peft_model(model, config)
elif fine_tuning_method == "freeze-subset":
    raise ValueError("To be implemented")
elif fine_tuning_method == "full-finetuning":
    pass

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir = run_name
    evaluation_strategy = EVALUATION_STRATEGY,
    learning_rate = LEARNING_RATE,
    weight_decay = WEIGHT_DECAY,
    num_train_epochs = NUM_TRAIN_EPOCHS,
    # save_strategy = SAVE_STRATEGY,
    run_name = RUN_NAME,
    report_to=["wandb"],
    # push_to_hub=True,
)
    
