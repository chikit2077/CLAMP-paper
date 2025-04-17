import os
os.environ['HF_HOME'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
import warnings

warnings.filterwarnings('ignore')

from model import PairClassificationModel
from utils import get_attention_dna, get_real_score, special_dict_to_map_bw
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

import time
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, BertConfig
import sklearn.metrics as metrics
import json
import gc
import csv
from peft import get_peft_config, get_peft_model, IA3Config, TaskType
from scipy.stats import pearsonr
import pyBigWig
import pyfaidx
from tqdm import tqdm  

accelerator = Accelerator()
device = accelerator.device

def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
def seq2kmer_single(seq, k):

    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def find_peak_summit(bw, chrom, start, end):
    
    if SPECIES in ["caenorhabditis_elegans", "triticum_aestivum"]:
        
        chrom = chrom.replace('chr', '')
        
        if chrom in special_dict_to_map_bw[SPECIES]:
            chrom = special_dict_to_map_bw[SPECIES][chrom]
        else:
            raise ValueError(f"Unknown chromosome name: {chrom}")
    try:
        signal = bw.values(chrom, start, end, numpy=True)
        peak_index = np.argmax(signal)
        return start + peak_index
    except Exception as e:
        
        
        return start + (end - start) // 2

def extract_sequence_with_boundary(fasta, chrom, start, end, center, seq_length=1500):
    
    if 'chr' not in list(fasta.keys())[0]:
        chrom = chrom.replace('chr', '')
    
    half_length = seq_length // 2
    extract_start = max(center - half_length, start)  
    extract_end = extract_start + seq_length

    
    if extract_end > end:
        extract_end = end
        extract_start = max(end - seq_length, start)

    
    sequence = fasta[chrom][extract_start:extract_end].seq


    return extract_start, extract_end, sequence



def extract_signal_features(bw, chrom, start, end, bins=64):
    
    if 'chr' not in list(bw.chroms())[0]:
        chrom = chrom.replace('chr', '')
    
    
    if SPECIES in ["caenorhabditis_elegans", "triticum_aestivum"]:
        
        chrom = chrom.replace('chr', '')
        
        if chrom in special_dict_to_map_bw[SPECIES]:
            chrom = special_dict_to_map_bw[SPECIES][chrom]
        else:
            raise ValueError(f"Unknown chromosome name: {chrom}")
        
    
    length = end - start
    bin_size = length // bins
    
    
    features = []
    for i in range(bins):
        bin_start = start + i * bin_size
        bin_end = bin_start + bin_size
        default = np.float32(0.001)
        try:
            signal = bw.values(chrom, bin_start, bin_end, numpy=True)
        except Exception as e:
            
            signal = default
        
        if signal is None or np.all(np.isnan(signal)):
            features.append(default)  
        else:
            
            valid_signal = signal[~np.isnan(signal)]
            mean_value = np.mean(valid_signal) if len(valid_signal) > 0 else default
            features.append(mean_value)

    return np.array(features)

class PairDataset(Dataset):
    def __init__(
        self,
        loop_data,
        signal_data,
        reference_genome,
        tokenizer
        ):
        self.loop_data = pd.read_csv(loop_data)
        
        if 'pet_count' in self.loop_data.columns and 'label' in self.loop_data.columns:
            self.loop_data = self.loop_data.drop(columns=['pet_count', 'label'])
        self.signal_data_path = signal_data  
        self._signal_data = None 
        self.reference_genome = pyfaidx.Fasta(reference_genome)
        self.tokenizer = tokenizer
        
    @property
    def signal_data(self):
        
        if self._signal_data is None:
            self._signal_data = pyBigWig.open(self.signal_data_path)
        return self._signal_data
    
    def __len__(self):
        return len(self.loop_data)

    def __getitem__(self, idx):
        chrom1, start1, end1, chrom2, start2, end2 = self.loop_data.iloc[idx]
        
        
        chrom1 = str(chrom1)
        chrom2 = str(chrom2)
        
        
        has_chr_prefix = 'chr' in list(self.signal_data.chroms())[0]
        chrom1 = ('chr' + chrom1.replace('chr', '')) if has_chr_prefix else chrom1.replace('chr', '')
        chrom2 = ('chr' + chrom2.replace('chr', '')) if has_chr_prefix else chrom2.replace('chr', '')

        
        summit_index_1 = find_peak_summit(self.signal_data, chrom1, start1, end1)
        summit_index_2 = find_peak_summit(self.signal_data, chrom2, start2, end2)
        
        seq_scope = self.tokenizer.model_max_length 
        extract_start_1, extract_end_1, seq1 = extract_sequence_with_boundary(self.reference_genome, chrom1, start1, end1, summit_index_1, seq_scope)
        extract_start_2, extract_end_2, seq2 = extract_sequence_with_boundary(self.reference_genome, chrom2, start2, end2, summit_index_2, seq_scope)
        
        signal_features_1 = extract_signal_features(self.signal_data, chrom1, extract_start_1, extract_end_1)
        signal_features_2 = extract_signal_features(self.signal_data, chrom2, extract_start_2, extract_end_2)
        
        distance = abs(summit_index_2 - summit_index_1)
        return seq1, seq2, signal_features_1, signal_features_2, distance
    
    def __del__(self):
        
        if hasattr(self, '_signal_data') and self._signal_data is not None:
            self._signal_data.close()
    
def collate_fn(batch, tokenizer):
    seqs1, seqs2, signal_features_1, signal_features_2, distances = zip(*batch)
    
    
    seqs1 = [" ".join(seq2kmer_single(seq, 6)) for seq in seqs1]
    seqs2 = [" ".join(seq2kmer_single(seq, 6)) for seq in seqs2]
    
    
    tokens1 = tokenizer(
        seqs1,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    tokens2 = tokenizer(
        seqs2,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    
    
    signal_features_1 = torch.tensor(np.array(signal_features_1))
    signal_features_2 = torch.tensor(np.array(signal_features_2))
    distances = torch.tensor(np.array(distances))
    return tokens1["input_ids"], tokens2["input_ids"], signal_features_1, signal_features_2, distances



def test_model(model, test_dataloader, tokenizer, device, test_result_dir):
    model_to_load = model
    try:
        collect_out = []
        model_to_load.eval()
        with torch.inference_mode():
            
            progress_bar = tqdm(test_dataloader, desc="Predicting", position=0, leave=True)
            for seq1, seq2, signal_features_1, signal_features_2, distances in progress_bar:
                import time
                start_time = time.time()  
                outputs = model_to_load(seq1, seq2, signal_features_1, signal_features_2, distances)
                end_time = time.time()  
                processing_time = end_time - start_time  
                print(f"Time: {processing_time:.4f} seconds")  
                
                
                gathered_outputs = accelerator.gather(outputs)
                
                collect_out += gathered_outputs.flatten().cpu().detach().numpy().tolist()
                
        
        
        if accelerator.is_local_main_process:
            
            original_data = test_dataloader.dataset.loop_data.copy()
            
            
            if len(collect_out) != len(original_data):
                accelerator.print(f"Warning: The number of prediction results ({len(collect_out)}) does not match the number of original data ({len(original_data)})")
                
                if len(collect_out) > len(original_data):
                    collect_out = collect_out[:len(original_data)]
                
                else:
                    collect_out.extend([0.0] * (len(original_data) - len(collect_out)))
            
            
            original_data['prediction_probability'] = collect_out
            
            original_data['prediction_label'] = (original_data['prediction_probability'] >= 0.5).astype(int)
            
            
            output_csv_path = os.path.join(test_result_dir, "prediction_results.csv")
            original_data.to_csv(output_csv_path, index=False)
            accelerator.print(f"Prediction results saved to {output_csv_path}")

        return True

    except Exception as e:
        accelerator.print(f"An error occurred during testing: {str(e)}")
        return False



def main():

    set_seeds(42)

    params = argparse.ArgumentParser(description="Loop prediction")

    params.add_argument(
        "--seed",
        required=False,
        default=42,
        type=int,
        help="The seed of the random number generator",
    )
    params.add_argument(
        "--base_model_path",
        help="The path of the base model",
        default='chikit2077/CLAMP-6mer-1500bp-pretrain',
        type=str
    )
    params.add_argument(
        "--clamp_model_path",
        help="The path of the CLAMP model",
        default='chikit2077/CLAMP',
        type=str
    )
    params.add_argument(
        "--batch_size",
        help="batch size",
        default=256,
        type=int
    )
    params.add_argument(
        "--result_dir",
        required=False,
        type=str,
        help="The path of the result directory",
    )
    params.add_argument(
        "--test",
        required=True,
        type=str,
        help="The path of the test set file",
    )
    params.add_argument(
        "--kmer",
        required=False,
        default=6,
        type=int,
        help="K-mer",
    )
    
    params.add_argument(
        "--bw",
        required=False,
        type=str,
        help="The path of the signal data file",
    )
    
    params.add_argument(
        "--reference_genome",
        required=False,
        type=str,
        help="The path of the reference genome file",
    )
    
    params.add_argument(
        "--species",
        required=False,
        type=str,
        help="Species",
    )

    args = params.parse_args()
    
    
    set_seeds(args.seed)
    
    global SPECIES
    SPECIES = args.species
    
    kmer = args.kmer

    batch_size = args.batch_size
    reference_genome = args.reference_genome
    bw_file = args.bw
    test_file = args.test
    result_dir = args.result_dir

    os.makedirs(result_dir, exist_ok=True)

    base_model = AutoModel.from_pretrained(args.base_model_path, add_pooling_layer=False)
    tokenizer = AutoTokenizer.from_pretrained(args.clamp_model_path)
    model = PairClassificationModel.from_pretrained(args.clamp_model_path, base_model=base_model)

    test_dataset = PairDataset(test_file, bw_file, reference_genome, tokenizer)
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=False,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )
    
    
    test_dataloader, model = accelerator.prepare(test_dataloader, model)
    
    test_success = test_model(
        model, test_dataloader, tokenizer, device, result_dir
    )

if __name__ == "__main__":
    
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time: {end_time - start_time} seconds")



