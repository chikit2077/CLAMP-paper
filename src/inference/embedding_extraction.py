import os
os.environ['HF_HOME'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
import warnings

warnings.filterwarnings('ignore')


from utils import get_attention_dna, get_real_score
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import random
import argparse
from Bio import SeqIO

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import time


def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def seq2kmer_single(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def visualize_pca(embedding, output_path, index):
    
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    
    
    pca = PCA(n_components=2)
    reduced_embedding = pca.fit_transform(embedding)
    
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], alpha=0.7)
    plt.title(f"PCA Visualization of {index}")
    plt.xlabel(f"Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.4f})")
    plt.ylabel(f"Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.4f})")
    plt.colorbar(plt.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], 
                              c=range(len(reduced_embedding)), cmap='viridis'), 
                 label='Token Position')
    plt.grid(alpha=0.3)
    
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    

def main():

    set_seeds(42)

    params = argparse.ArgumentParser(description="Embedding extraction")

    params.add_argument(
        "--base_model_path",
        required=False,
        default="chikit2077/CLAMP-4mer-500bp-pretrain",
        type=str,
        help="The path of the pretrained model",
    )
    params.add_argument(
        "--seed",
        required=False,
        default=42,
        type=int,
        help="The seed of the random number generator",
    )
    params.add_argument(
        "--tokenizer_path",
        required=False,
        default="chikit2077/CLAMP-4mer-500bp-pretrain",
        type=str,
        help="The path of the tokenizer (k-mer)",
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
        "--kmer",
        required=False,
        default=4,
        type=int,
        help="K-mer",
    )
    params.add_argument(
        "--fasta_file",
        required=True,
        type=str,
        help="The path of the fasta file",
    )

    args = params.parse_args()
    
    
    set_seeds(args.seed)
    

    
    pretrained_model_path = args.base_model_path
    tokenizer_path = args.tokenizer_path
    kmer = args.kmer

    batch_size = args.batch_size
    fasta_file = args.fasta_file
    
    result_dir = args.result_dir

    os.makedirs(result_dir, exist_ok=True)
    
    pca_vis_dir = os.path.join(result_dir, "pca_visualization")
    os.makedirs(pca_vis_dir, exist_ok=True)
        
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    atac_model = AutoModel.from_pretrained(
        pretrained_model_path, output_attentions=True, add_pooling_layer=False,
        attn_implementation="eager"
    )

    
    fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')
    fasta_list = []
    sequence_ids = []  
    seq_length_list = []
    for fasta in fasta_sequences:
        sequence = str(fasta.seq).upper()
        fasta_list.append(sequence)
        
        seq_length_list.append(len(sequence)-kmer+1)
    sequence_ids = [f"Sequence {i}" for i in range(len(fasta_list))]
    inputs = tokenizer(
        [" ".join(seq2kmer_single(seq, kmer)) for seq in fasta_list], 
        return_tensors="pt", 
        padding="max_length",
        truncation=True
    )

    with torch.inference_mode():
        model_output = atac_model(inputs['input_ids'])  
        hidden_states = model_output['last_hidden_state'] 
        attentions = model_output['attentions'] 
    
    np.save(os.path.join(result_dir, "Embeddings.npy"), hidden_states)
    attns = get_attention_dna(attentions, 11, 11, seq_length_list)

    
    os.makedirs(os.path.join(result_dir, 'attention'), exist_ok=True)
    for sample_idx in range(len(attns)):
        attention_scores = np.array(attns[sample_idx]).reshape(-1,1)
        real_scores = get_real_score(attention_scores, 4, "mean")
        scores = real_scores.reshape(1, -1)
        scores_df = pd.DataFrame({"nucleotide": list(fasta_list[sample_idx]), "scores": scores[0]})
        scores_df.to_csv(os.path.join(result_dir, 'attention', f"attention_weights_{sequence_ids[sample_idx]}.csv"), index=False)

    
    
    for i in range(hidden_states.shape[0]):
        
        seq_embedding = hidden_states[i]  
        
        
        if i < len(sequence_ids):
            seq_name = sequence_ids[i]
        else:
            seq_name = f"sequence_{i}"
        
        output_path = os.path.join(pca_vis_dir, f"pca_vis_{seq_name}.png")
        
        
        visualize_pca(seq_embedding, output_path, seq_name)
    


if __name__ == "__main__":
    
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time: {end_time - start_time} seconds")



