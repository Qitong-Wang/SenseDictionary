import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import argparse
import random
import pickle
from tqdm import tqdm
import numpy as np
import json
from collections import OrderedDict
from DeBERTa.deberta.spm_tokenizer import SPMTokenizer

import pickle


def _truncate_segments(segments, max_num_tokens=128):
  """
  Truncate sequence pair according to original BERT implementation:
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
  """
  while True:
    if sum(len(s) for s in segments)<=max_num_tokens:
      break

    segments = sorted(segments, key=lambda s:len(s), reverse=True)
    trunc_tokens = segments[0]

    assert len(trunc_tokens) >= 1
    trunc_tokens.pop()
  return segments 



class TokenClassificationDataset(Dataset):
    def __init__(self, filepath, tokenizer, input_length=128):
        with open(filepath, 'r', encoding='utf-8') as file:
            self.pairs  = json.load(file)
        self.tokenizer = tokenizer
        self.input_length = input_length
        print("input length:", self.input_length)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx): # GLUE for Deberta

        
        prompts = self.pairs[idx]
        segments = _truncate_segments([self.tokenizer.tokenize(s) for s in prompts], self.input_length)
        tokens = ['[CLS]']
        for i,s in enumerate(segments):
            tokens.extend(s)
            tokens.append('[SEP]')
            

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(token_ids)
        features = OrderedDict(input_ids = token_ids, input_mask = input_mask)

        
        return features


def pad_tensors(lists, pad_value=0):

    max_length = max(len(sublist) for sublist in lists)
    
    # Pad each sublist to the maximum length
    padded_lists = [sublist + [pad_value] * (max_length - len(sublist)) for sublist in lists]
    padded_lists = torch.tensor(padded_lists)
    return padded_lists
def collate_fn(batch):
    input_ids = []
    input_mask = []
    for item in batch:
        if isinstance(item, dict):
            input_ids.append(item['input_ids'])
            input_mask.append(item['input_mask'])
        else:
            continue

    input_ids_padded = pad_tensors(input_ids)
    input_mask_padded = pad_tensors(input_mask)
    if input_ids_padded.shape[1] > 512:
        input_ids_padded = input_ids_padded[:,:512]
        input_mask_padded = input_mask_padded[:,:512]


    return {
        'input_ids': input_ids_padded,
        'input_mask' : input_mask_padded

    }
    

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, help="input json file")
    parser.add_argument('--output_file', type=str, help="output json file")
    parser.add_argument('--input_length', type=int, help="input corpus length")
    args = parser.parse_args()


    vocab_size = 128000

    # Initialize  the DataLoader
    tokenizer = SPMTokenizer( "./cache/assets/latest/deberta-v3-large/spm.model")
    
    count_dict  = dict()
    

    dataset = TokenClassificationDataset(args.json_file, tokenizer, args.input_length)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn, num_workers=4)

    stop_training = False
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids']
            input_mask = batch['input_mask']
            for i in range(input_ids.shape[0]):
                for j in range(input_ids.shape[1]):
                    if input_mask[i][j]  == 0:
                        continue
                    id = input_ids[i][j].item()
                    count_dict.setdefault(id, 0)
                    count_dict[id] += 1

    print("Save to",args.output_file)
    with open(args.output_file, 'wb') as handle:
        pickle.dump(count_dict, handle)
