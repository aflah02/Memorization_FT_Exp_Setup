import numpy as np
import random
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import (get_peft_model, LoraConfig, TaskType, IA3Config, 
                  PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, PromptTuningInit,
                  IA3Model)
import torch
from custom_datasets import *

EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "never"

def run_training(params_dict):
    seed = params_dict["seed"]
    task_name = params_dict["task"]
    model_id = params_dict["model"]
    num_seq = params_dict["num_seq"]
    seq_len = params_dict["seq_len"]
    fine_tuning_method = params_dict["fine_tuning_method"]
    run_name = params_dict["wandb_run_name"]
    learning_rate = params_dict["learning_rate"]
    weight_decay = params_dict["weight_decay"]
    num_train_epochs = params_dict["num_train_epochs"]

    # Set the seed
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_id}")
    tokenizer.pad_token = tokenizer.eos_token
    if task_name == "random_strings":
        dataset = generate_random_string_dataset(seed=seed, num_sequences=num_seq, sequence_length=seq_len)
        dataset = encode_character_wise(tokenizer, dataset)
    else:
        raise ValueError("Invalid task name")

    model_id = 'EleutherAI/' + model_id

    # Get the model
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Get the config
    if fine_tuning_method == "IA3":
        target_modules_IA3 = params_dict["target_modules_IA3"]
        feed_forward_modules_IA3 = params_dict["feed_forward_modules_IA3"]
        config = IA3Config(peft_type="IA3", task_type=TaskType.CAUSAL_LM, target_modules=target_modules_IA3, feed_forward_modules=feed_forward_modules_IA3)
        model = IA3Model(model, config)
    elif fine_tuning_method == "lora":
        r_lora = params_dict["r_lora"]
        alpha_lora = params_dict["alpha_lora"]
        dropout_lora = params_dict["dropout_lora"]
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r_lora, alpha=alpha_lora, dropout=dropout_lora, inference_mode=False)
        model = get_peft_model(model, config)
    elif fine_tuning_method == "prefix-tuning":
        num_virtual_tokens = params_dict["num_virtual_tokens"]
        config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens, inference_mode=False)
        model = get_peft_model(model, config)
    elif fine_tuning_method == "prompt-tuning":
        num_virtual_tokens = params_dict["num_virtual_tokens"]
        init_text_prompt_tuning = params_dict["init_text_prompt_tuning"]
        config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens, prompt_tuning_init=PromptTuningInit.TEXT, init_text=init_text_prompt_tuning, tokenizer_name_or_path=model_id)
        model = get_peft_model(model, config)
    elif fine_tuning_method == "p-tuning":
        num_virtual_tokens = params_dict["num_virtual_tokens"]
        encoder_hidden_size_p_tuning = params_dict["encoder_hidden_size_p_tuning"]
        config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens, encoder_hidden_size=encoder_hidden_size_p_tuning)
        model = get_peft_model(model, config)
    elif fine_tuning_method == "freeze-subset":
        bias_trainable = params_dict["bias_trainable"]
        layer_norm_trainable = params_dict["layer_norm_trainable"]
        mlp_trainable = params_dict["mlp_trainable"]
        attention_trainable = params_dict["attention_trainable"]
        layers_trainable = params_dict["layers_trainable"]
        embed_in_trainable = params_dict["embed_in_trainable"]
        embed_out_trainable = params_dict["embed_out_trainable"]
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = bias_trainable
            elif "layernorm" in name:
                param.requires_grad = layer_norm_trainable
            elif ".mlp." in name:
                param.requires_grad = mlp_trainable
            elif ".attention." in name:
                param.requires_grad = attention_trainable
            elif "layers." in name:
                layer_num = int(name.split(".")[1])
                if layer_num in layers_trainable:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif "embed_in" in name:
                param.requires_grad = embed_in_trainable
            elif "embed_out" in name:
                param.requires_grad = embed_out_trainable
            else:
                raise ValueError("Invalid parameter name")
    elif fine_tuning_method == "full-finetuning":
        pass

    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir = run_name,
        evaluation_strategy = EVALUATION_STRATEGY,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        num_train_epochs = num_train_epochs,
        save_strategy = SAVE_STRATEGY,
        run_name = run_name,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()