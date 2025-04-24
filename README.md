import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
 # 用于计算 FLOPs
import time
class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True,
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_train_dataset = [dp for dp in self.local_train_dataset if dp is not None]
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = [dp for dp in self.local_eval_dataset if dp is not None]
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_train_dataset = [dp for dp in self.local_train_dataset if dp is not None]
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size
    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp,miu):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            #optim="adamw_torch",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        
        )

        # 自定义优化器，只对 LoRA 可训练参数应用 L2 正则化
        lora_params = [p for n, p in self.model.named_parameters() if "lora" in n.lower() and p.requires_grad]
        param_groups = [
            {"params": lora_params, "weight_decay": self.train_args.weight_decay},
        ]
        optimizer = torch.optim.Adam(param_groups, lr=local_learning_rate,weight_decay=miu)
        # optimizer = torch.optim.SGD(param_groups, lr=local_learning_rate)

        class CustomTrainer(transformers.Trainer):
            def training_step(self, model, inputs,m):

                loss = super().training_step(model, inputs)
                # if miu!=0:
                #     l2_reg = torch.tensor(0., requires_grad=True).to(model.device)
                #     for param in lora_params:
                #         l2_reg = l2_reg + torch.norm(param, p=2) ** 2
                # # 添加正则化项到损失中
                #     loss = loss + 0.5 * miu * l2_reg
                # 计算 L2 正则化损失
                
                return loss
            # def training_epoch_end(self, outputs):
            #     self.scaler.step(self.optimizer)  # 更新优化器
            #     self.scaler.update()  # 更新 GradScaler
                
        self.local_trainer = CustomTrainer(model=self.model,
                                           train_dataset=self.local_train_dataset,
                                           eval_dataset=self.local_eval_dataset,
                                           args=self.train_args,
                                           data_collator=transformers.DataCollatorForSeq2Seq(
                                               tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                           ),
                                           optimizers=(optimizer, None)
                                           )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        start_time=time.time()
        self.local_trainer.train()
        end_time=time.time()
        training_time = end_time - start_time
        with open("log_uvar.txt", 'a') as f:
            f.write(f"Client {self.client_id} training time: {training_time} seconds\n")

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id 

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
