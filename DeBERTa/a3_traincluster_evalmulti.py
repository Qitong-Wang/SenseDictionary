import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel, DebertaV2Model
import numpy as np
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle
import argparse
import einops
import os
from ranger_adabelief import RangerAdaBelief  # Import the Ranger optimizer
from a4_deberta_models import  TeacherModel, StudentModel
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertModel, BertTokenizer
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import get_scheduler
import time
def pad2d_withmask(tensor_list):
    # Determine the maximum size of the first dimension across all tensors
    max_size = max(t.shape[0] for t in tensor_list)
    num_tensors = len(tensor_list)
    feature_dim = tensor_list[0].shape[1]

    # Initialize the padded tensor with zeros
    padded_tensor = torch.zeros(num_tensors, max_size, feature_dim)

    # Initialize the mask with zeros (indicating padding)
    mask = torch.zeros(num_tensors, max_size, dtype=torch.bool)
    # Pad each tensor and update the mask
    for i, tensor in enumerate(tensor_list):
        padded_tensor[i, :tensor.shape[0],:tensor.shape[1]] = tensor
        mask[i, :tensor.shape[0]] = 1

    return padded_tensor, mask



def pad1d(tensor_list):
    padded_tensors_tensor = pad_sequence(tensor_list, batch_first=True, padding_value=0)
    return padded_tensors_tensor


def pad2d(tensors, padding_value=0):
    # Determine the max height and width
    max_height = max(tensor.size(0) for tensor in tensors)
    max_width = max(tensor.size(1) for tensor in tensors)

    # Pad each tensor to the same size
    padded_tensors = []
    for tensor in tensors:
        padding = (0, max_width - tensor.size(1), 0, max_height - tensor.size(0))
        padded_tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors into a single tensor
    padded_tensors_tensor = torch.stack(padded_tensors)

    return padded_tensors_tensor

def pad3d(tensors, padding_value=0):
    # Determine the max depth, height, and width
    max_depth = max(tensor.size(0) for tensor in tensors)
    max_height = max(tensor.size(1) for tensor in tensors)
    max_width = max(tensor.size(2) for tensor in tensors)

    # Pad each tensor to the same size
    padded_tensors = []
    for tensor in tensors:
        padding = (0, max_width - tensor.size(2), 0, max_height - tensor.size(1), 0, max_depth - tensor.size(0))
        padded_tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        padded_tensors.append(padded_tensor)

    padded_tensors_tensor = torch.stack(padded_tensors)
    return padded_tensors_tensor


def custom_collate_fn(batch):
    transposed = list(zip(*batch))
    padded_tensors = []

    for tensors in transposed:
        if tensors[0].dim() == 1:
            padded_tensors.append(pad1d(tensors))
        elif tensors[0].dim() == 2:
            padded_tensors.append(pad2d(tensors))
        elif tensors[0].dim() == 3:
            padded_tensors.append(pad3d(tensors))
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(tensors[0].dim()))

    return tuple(padded_tensors)

class TokenClassificationDataset(Dataset):
    def __init__(self, tokenizer,json_path,cluster_path):
        self.tokenizer = tokenizer
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        path = cluster_path
        with open(path, 'rb') as handle:
            self.token_emb_dict = pickle.load(handle)
        self.omit_token = []

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        segments = self.data[idx]
        segments = [tokenizer.tokenize(s) for s in segments]
        tokens = ['[CLS]']
        type_ids = [0]
        for i,s in enumerate(segments):
            tokens.extend(s)
            tokens.append('[SEP]')
            type_ids.extend([i]*(len(s)+1))
        tokens = tokens[:512]
        type_ids = type_ids[:512]
        
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(token_ids)
        pos_ids = list(range(len(token_ids)))
        token_ids = torch.tensor(token_ids)
        input_mask = torch.tensor(input_mask)
        type_ids = torch.tensor(type_ids)
        pos_ids = torch.tensor(pos_ids)

      
        # Tokenizing the sentence
        cluster_mean_list = []
        cluster_filter_list = []
        for i in range(token_ids.shape[0]):
            element = token_ids[i].item()
            if element == 2 or element == 1: #SEP and CLS
                cluster_mean_list.append(torch.zeros(1,1024).half())
                cluster_filter_list.append(0)
            elif element in self.omit_token or element not in self.token_emb_dict or self.token_emb_dict[element] is None  or len(self.token_emb_dict[element]) == 0  :
                cluster_mean_list.append(torch.zeros(1,1024).half())
                cluster_filter_list.append(0)
            else:
                cluster_mean_list.append(self.token_emb_dict[element])
                cluster_filter_list.append(1)

        cluster_mean, cluster_padding_filter = pad2d_withmask(cluster_mean_list)
        cluster_fiter = torch.tensor(cluster_filter_list).bool()
        
        return token_ids,input_mask,type_ids,pos_ids, cluster_mean.half(), cluster_padding_filter.bool(),cluster_fiter



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=2, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--ckpt_path', type=str, help='Student ckpt path')
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='Path for continue student learning')
    parser.add_argument('--model_config', type=str, default="./experiments/glue/config.json", help='Model Configuration')
    parser.add_argument('--json_path', type=str, help='Input dataset path')
    parser.add_argument('--cluster_path', type=str, help='Sense dictionary Path')
    parser.add_argument('--teacher_ckpt_path', type=str, help='Teacher ckpt path')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of Label')
    parser.add_argument('--fp16', type=bool, default=True, help='FP16 or not')
    parser.add_argument('--warmup', type=float, default=0, help='Warnup for scheduler')
    parser.add_argument('--scheduler', type=bool, default=True, help='Whether use scheduler')
    args = parser.parse_args()


    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    # Tokenization and dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)
    dataset = TokenClassificationDataset( tokenizer, cluster_path=args.cluster_path, json_path=args.json_path)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank,shuffle=True)
    # DataLoader
    data_loader = DataLoader(dataset, batch_size=32, sampler=sampler, collate_fn=custom_collate_fn)


    teacher_model = TeacherModel.load_model( "deberta-v3-large", args.model_config, num_labels=args.num_labels, drop_out=0.0,cluster_path=None, only_return_hidden=True )
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("Load ckpt")
   
    model_state_dict = torch.load(args.teacher_ckpt_path, map_location=torch.device('cpu'))
    teacher_model.load_state_dict(model_state_dict)
    teacher_model = teacher_model.to(local_rank)
    if args.fp16:
       teacher_model = teacher_model.half() #fp16


    model = StudentModel.load_model( "deberta-v3-xsmall", args.model_config, num_labels=args.num_labels, drop_out=0.0,cluster_path=None, only_return_hidden=True)

    copied_parameters = []
    parameters_to_copy = ["pooler", "classifier"]  # Specify which parameters to copy

    for name_A, param_A in model.named_parameters():
        # Check if the parameter name contains any of the specified substrings
        if any(target in name_A for target in parameters_to_copy) and name_A in teacher_model.state_dict():
            param_B = teacher_model.state_dict()[name_A]
            if param_A.shape == param_B.shape:
                param_A.data.copy_(param_B.data)
                copied_parameters.append(name_A)

    print("Copied parameters:", copied_parameters)

    model.deberta.embeddings.position_embeddings.weight.requires_grad = False
    model.pooler.dense.weight.requires_grad = False
    model.pooler.dense.bias.requires_grad = False
    model.classifier.weight.requires_grad = False
    model.classifier.bias.requires_grad = False

    model.adaptor.weight.requires_grad = True
    model.adaptor.bias.requires_grad = True
    if args.fp16:
       model = model.half() #fp16   
    
        

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')

    start_epoch = 0
    start_batch_idx = 0
    if args.load_ckpt_path:
        print("Student load model")
        model_state_dict = torch.load(args.load_ckpt_path, map_location=torch.device('cpu'))

        new_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
        optim_checkpoint = torch.load(args.load_ckpt_path[:-5]+".optimckpt", map_location=torch.device('cpu'))
        start_epoch = optim_checkpoint['epoch'] + 1
        start_batch_idx = optim_checkpoint['batch_idx']
        
    
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

 
    optimizer = RangerAdaBelief(model.parameters(), lr=args.lr,betas=(0.9,0.999),eps=1e-8,weight_decay=5e-4,weight_decouple=True)


    if args.load_ckpt_path:
        optimizer.load_state_dict(optim_checkpoint['optimizer_state_dict'])

    import math
    num_training_steps =math.ceil(len(dataset) / (dist.get_world_size() * 32)) * args.epoch
    num_warmup_steps = int (0.05 * num_training_steps) #int(0.1 * num_training_steps ) #int(0.1 * num_training_steps)  
    
    # warmup scheduler
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr*1.2,  
        total_steps=num_training_steps,
        pct_start=args.warmup,  
        anneal_strategy="cos", 
        final_div_factor=10 
        )
    if args.load_ckpt_path:
        if args.scheduler:
            scheduler.load_state_dict(optim_checkpoint['scheduler_state_dict'])

    save_interval = 3300  
    last_save_time = time.time()  # Record the starting time
    for epoch in tqdm(range(start_epoch,args.epoch)):
        sampler.set_epoch(epoch)
        print(f"epoch: {epoch}")
        total_correct =0
        total_count = 0
        wrong_id_dict = dict()
     
        # Iterating through data in the dataloader
        for batch_idx, (token_ids,input_mask,type_ids,pos_ids, cluster_mean, cluster_padding_filter,cluster_fiter) in enumerate(tqdm(data_loader)):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue  # Skip already processed batches
            optimizer.zero_grad(set_to_none=True)
            input_ids = token_ids.to(local_rank)
            attention_mask = input_mask.to(local_rank)
            token_type_ids = type_ids.to(local_rank)
            position_ids = pos_ids.to(local_rank)
            cluster_mean = cluster_mean[cluster_fiter].to(local_rank)
            cluster_padding_filter = cluster_padding_filter[cluster_fiter].to(local_rank)
            cluster_fiter = cluster_fiter.to(local_rank)
            

            mapped_outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, output_all_encoded_layers=True,rank=local_rank)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    position_ids=position_ids, output_all_encoded_layers=True,rank=local_rank)
            batch_loss = 0
            batch_correct = 0
            batch_count = 0



            mapped_outputs = mapped_outputs[cluster_fiter]
            teacher_outputs = teacher_outputs[cluster_fiter]
           
            mapped_outputs = mapped_outputs.unsqueeze(2)  
            teacher_outputs = teacher_outputs.unsqueeze(2)

           
            map_logits = torch.einsum('bij,bjk->bik',cluster_mean, mapped_outputs )
            map_logits = map_logits[:,:,0]  


            teacher_logits = torch.einsum('bij,bjk->bik', cluster_mean, teacher_outputs)
            teacher_logits = teacher_logits[:,:,0] 
            
            map_logits[~cluster_padding_filter] = 0
            teacher_logits[~cluster_padding_filter] = 0 

    
            map_labels=torch.argmax(map_logits,dim=1)

          
            teacher_labels = torch.argmax(teacher_logits,dim=1)
          
            batch_count = map_labels.shape[0]
            batch_correct = torch.sum(map_labels==teacher_labels)
       
            
            batch_loss = loss_fn(map_logits, teacher_labels)  
            batch_loss_back = batch_loss/ batch_count

            batch_loss_back.backward()


            optimizer.step()
            if args.scheduler:
                scheduler.step()


            batch_loss_tensor = batch_loss.detach().clone()
            batch_count_tensor = torch.tensor(batch_count, dtype=torch.float32, device=batch_loss.device)
            batch_correct_tensor = torch.tensor(batch_correct, dtype=torch.float32, device=batch_loss.device)

            dist.all_reduce(batch_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_correct_tensor, op=dist.ReduceOp.SUM)

            # Normalize for logging only (do not backprop again)
            batch_loss_mean = batch_loss_tensor / batch_count_tensor
            batch_correct_mean = batch_correct_tensor / batch_count_tensor


            
            if rank == 0 and batch_idx % 100 == 0:
                print("step loss: {:.2f}, step_acc: {:.2f}".format(batch_loss_mean, batch_correct_mean))

            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                last_save_time = current_time
                if rank == 0:
                    # Save the model
                    
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(), 
                    }
                    torch.save(checkpoint, args.ckpt_path[:-5]+time.strftime("%Y%m%d_%H%M%S")+".optimckpt")
                    torch.save(model.state_dict(), args.ckpt_path[:-5]+time.strftime("%Y%m%d_%H%M%S")+".ckpt")
                    print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")

                torch.distributed.barrier()

            
        if rank == 0:
            print("step loss: {:.2f}, step_acc: {:.2f}".format(batch_loss_mean, batch_correct_mean))
            torch.save(model.state_dict(), args.ckpt_path[:-5]+str(epoch)+".ckpt")
            
            checkpoint = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),

                }
            torch.save(checkpoint, args.ckpt_path[:-5]+str(epoch)+".optimckpt")
        torch.distributed.barrier()
            
    dist.destroy_process_group()