import sys


import torch
import numpy as np

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    
    return torch.stack(squeezed)

def get_attention_dna(attention, start, end, seq_length_list):
    attn = format_attention(attention).transpose(1, 0) 
    result_list = []
    for batch_size in range(attn.shape[0]):
        attn_score = []
        
        for i in range(1, seq_length_list[batch_size] + 1):
            
            attn_score.append(float(attn[batch_size, start:end + 1, :, 0, i].sum()))
            
        result_list.append(attn_score)
    return result_list

def get_real_score(attention_scores, kmer, metric):
    counts = np.zeros([len(attention_scores) + kmer - 1])
    real_scores = np.zeros([len(attention_scores) + kmer - 1])
    
    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i + j] += 1.0
                real_scores[i + j] += score

        real_scores = real_scores / counts
    else:
        pass

    return real_scores
    
special_dict_to_map_bw = {
    "caenorhabditis_elegans": {
        "NC_003279.8": "chrI",
        "NC_003280.10": "chrII",
        "NC_003281.10": "chrIII",
        "NC_003282.8": "chrIV",
        "NC_003283.11": "chrV",
        "NC_003284.9": "chrX",
    },
    "triticum_aestivum": {
        "NC_057794.1": "1A",
        "NC_057795.1": "1B",
        "NC_057796.1": "1D",
        "NC_057797.1": "2A",
        "NC_057798.1": "2B",
        "NC_057799.1": "2D",
        "NC_057800.1": "3A",
        "NC_057801.1": "3B",
        "NC_057802.1": "3D",
        "NC_057803.1": "4A",
        "NC_057804.1": "4B",
        "NC_057805.1": "4D",
        "NC_057806.1": "5A",
        "NC_057807.1": "5B",
        "NC_057808.1": "5D",
        "NC_057809.1": "6A",
        "NC_057810.1": "6B",
        "NC_057811.1": "6D",
        "NC_057812.1": "7A",
        "NC_057813.1": "7B",
        "NC_057814.1": "7D",
    }
}