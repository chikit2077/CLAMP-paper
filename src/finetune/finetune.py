import sys
import os
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, BertConfig
import pandas as pd
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np
import json
import random
import argparse
import gc
import csv
from peft import get_peft_config, get_peft_model, IA3Config, TaskType
from scipy.stats import pearsonr
import pyBigWig
import pyfaidx

accelerator = Accelerator()
device = accelerator.device

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
        chrom1, start1, end1, chrom2, start2, end2, pet_count, label = self.loop_data.iloc[idx]
        
        
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
        return seq1, seq2, signal_features_1, signal_features_2, distance, label
    
    def __del__(self):
        
        self.signal_data.close()
    
def collate_fn(batch, tokenizer):
    seqs1, seqs2, signal_features_1, signal_features_2, distances, labels = zip(*batch)
    
    
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
    labels = torch.tensor(np.array(labels))
    return tokens1["input_ids"], tokens2["input_ids"], signal_features_1, signal_features_2, distances, labels

def compute_metrics(label, score, evaluation_metric="auroc"):
    if evaluation_metric in ["acc", "recall", "precision", "f1", "kappa", "mcc"]:
        
        score = [1 if s >= 0.5 else 0 for s in score]
        
    if evaluation_metric == "auroc":
        performance = metrics.roc_auc_score(y_true=label, y_score=score)
    elif evaluation_metric == "auprc":
        performance = metrics.average_precision_score(y_true=label, y_score=score)
    elif evaluation_metric == "acc":
        performance = metrics.accuracy_score(label, score)
    elif evaluation_metric == "recall":
        performance = metrics.recall_score(label, score, average="binary")
    elif evaluation_metric == "precision":
        performance = metrics.precision_score(label, score, average="binary")
    elif evaluation_metric == "f1":
        performance = metrics.f1_score(label, score, average="binary")
    elif evaluation_metric == "kappa":
        performance = metrics.cohen_kappa_score(label, score)
    elif evaluation_metric == "mcc":
        performance = metrics.matthews_corrcoef(label, score)
    else:
        performance = None
    return performance



class PairClassificationModel(nn.Module):
    def __init__(self, base_model):
        super(PairClassificationModel, self).__init__()
        
        peft_config = IA3Config(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            target_modules=["key", "value", "output.dense"],
            feedforward_modules=["output.dense"],
        )
        self.base_model = get_peft_model(base_model, peft_config)
        accelerator.print(self.base_model.print_trainable_parameters())
        
        self.signal_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1)
        )

        
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 64)
        )

        
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 + 128, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        )

        
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),  
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, seq1, seq2, signal_features_1, signal_features_2, distance):
        
        seq1_embed = self.get_embedding(seq1)
        signal1_embed = self.signal_encoder(signal_features_1)
        region1_features = self.fusion_layer(torch.cat([seq1_embed, signal1_embed], dim=1))  

        
        seq2_embed = self.get_embedding(seq2)
        signal2_embed = self.signal_encoder(signal_features_2)
        region2_features = self.fusion_layer(torch.cat([seq2_embed, signal2_embed], dim=1))  

        
        regions = torch.stack([region1_features, region2_features], dim=0)  

        
        attn_output, attn_weights = self.cross_attention(regions, regions, regions)

        
        attn_output = attn_output.permute(1, 0, 2)  

        
        combined_region_features = attn_output.mean(dim=1)  

        
        log_distance = torch.log1p(distance)
        distance_features = self.distance_encoder(log_distance.unsqueeze(-1))  

        
        combined = torch.cat([combined_region_features, distance_features], dim=1)  
        output = self.classifier(combined)  
        return torch.sigmoid(output)

    def get_embedding(self, inputs):
        hidden_states = self.base_model(inputs)[0]  
        embedding = hidden_states[:, 0, :]  
        return embedding


def val_model(model, val_data, tokenizer, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0
    collect_y = []
    collect_out = []

    with torch.inference_mode():
        progress_bar = tqdm(
            val_data, desc="Validating", disable=not accelerator.is_local_main_process
        )
        for seq1, seq2, signal_features_1, signal_features_2, distances, labels in progress_bar:


            outputs = model(seq1, seq2, signal_features_1, signal_features_2, distances)
            loss = criterion(outputs.squeeze(), labels.float())

            
            gathered_loss = accelerator.gather(loss).mean().item()
            gathered_labels = accelerator.gather(labels)
            gathered_outputs = accelerator.gather(outputs)

            
            batch_size = gathered_labels.size(0)
            total_loss += gathered_loss * batch_size
            total_samples += batch_size

            collect_y += gathered_labels.cpu().tolist()
            collect_out += gathered_outputs.flatten().cpu().detach().numpy().tolist()

            
            progress_bar.set_postfix(loss=total_loss / total_samples)

        progress_bar.close()

    
    accelerator.wait_for_everyone()

    
    metrics_dict = {}
    metrics_dict["loss"] = total_loss / total_samples
    
    evaluation_metrics = [
        "auroc", "auprc", "acc", "recall", "precision", "f1", "kappa", "mcc"
    ]
    for metric in evaluation_metrics:
        metrics_dict[metric] = compute_metrics(
            collect_y, collect_out, evaluation_metric=metric
        )

    return metrics_dict


def test_model(model, test_dataloader, tokenizer, device, model_save_dir, test_result_dir):
    model_to_load = accelerator.unwrap_model(model)
    try:
        
        if os.path.exists(os.path.join(model_save_dir, "model_valbest.pth")):
            model_to_load.load_state_dict(
                torch.load(os.path.join(model_save_dir, "model_valbest.pth"), weights_only=True)
            )
        else:
            model_to_load.load_state_dict(
                torch.load(os.path.join(model_save_dir, "model_final.pth"), weights_only=True)
            )

        collect_y = []
        collect_out = []
        model_to_load.eval()
        with torch.inference_mode():
            progress_bar = tqdm(
                test_dataloader,
                desc="Testing",
                disable=not accelerator.is_local_main_process,
            )
            for seq1, seq2, signal_features_1, signal_features_2, distances, labels in progress_bar:

                outputs = model_to_load(seq1, seq2, signal_features_1, signal_features_2, distances)
                
                
                gathered_labels = accelerator.gather(labels)
                gathered_outputs = accelerator.gather(outputs)

                collect_y += gathered_labels.cpu().tolist()
                collect_out += gathered_outputs.flatten().cpu().detach().numpy().tolist()

            progress_bar.close()

        
        

        
        if accelerator.is_main_process:
            
            predictions_df = pd.DataFrame({
                'true_label': collect_y,
                'predicted_score': collect_out,
                'predicted_label': [1 if score >= 0.5 else 0 for score in collect_out]
            })
            predictions_df.to_csv(os.path.join(test_result_dir, 'predictions.csv'), index=False)
            
            
            evaluation_metric = ["acc", "recall", "precision", "f1", "kappa", "mcc", "auroc", "auprc"]
            metric_scores = {}
            for metric in evaluation_metric:
                metric_scores[metric] = compute_metrics(collect_y, collect_out, evaluation_metric=metric)
                
            
            performance_df = pd.DataFrame({
                'metric': evaluation_metric,
                'score': [metric_scores[metric] for metric in evaluation_metric]
            })
            
            
            performance_df.to_csv(os.path.join(test_result_dir, "performance.csv"), index=False)
            
            
            for metric in evaluation_metric:
                accelerator.print(f"{metric}: {metric_scores[metric]}")

        return True

    except Exception as e:
        accelerator.print(f"测试过程中发生错误: {str(e)}")
        return False


def finetune_model(model, train_dataloader, val_dataloader, test_dataloader, tokenizer, criterion, optim, scheduler, epochs, earlystopping_tolerance, save_dir):
    train_loss_epoch = []
    step_loss_collect = []
    val_loss_epoch = []
    val_mcc_epoch = []
    val_mcc_best = -1  
    earlystopping_watchdog = 0

    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}",
            disable=not accelerator.is_local_main_process,
        )
        model.train()
        
        for seq1, seq2, signal_features_1, signal_features_2, distances, labels in train_dataloader:

            outputs = model(seq1, seq2, signal_features_1, signal_features_2, distances)
            loss = criterion(outputs.squeeze(), labels.float())
            
            
            gathered_loss = accelerator.gather(loss).mean().item()
            
            accelerator.backward(loss)
            optim.step()
            optim.zero_grad()

            epoch_loss += gathered_loss
            step_loss_collect.append(gathered_loss)

            progress_bar.update(1)
            progress_bar.set_postfix(loss=epoch_loss / len(train_dataloader))
            
        progress_bar.close()
        
        
        accelerator.wait_for_everyone()
        
        train_loss_epoch.append(np.mean(np.array(step_loss_collect)))

        val_metrics = val_model(model, val_dataloader, tokenizer, criterion)
        val_loss = val_metrics["loss"]
        val_mcc = val_metrics["mcc"]  
        val_loss_epoch.append(val_loss)
        val_mcc_epoch.append(val_mcc)
        
        
        scheduler.step(val_mcc)
        
        metrics_str = f"epoch:{epoch+1}, train loss:{train_loss_epoch[-1]:.5f}, val loss:{val_loss:.5f}, val_mcc:{val_mcc:.5f}"
        for metric, value in val_metrics.items():
            if metric not in ["loss", "mcc"]:
                metrics_str += f", val_{metric}:{value:.5f}"
        accelerator.print(metrics_str)

        
        if val_mcc > val_mcc_best:
            earlystopping_watchdog = 0
            val_mcc_best = val_mcc
            
            model_to_save = accelerator.unwrap_model(model)
            accelerator.save(model_to_save.state_dict(), os.path.join(save_dir, "model_valbest.pth"))
            

        earlystopping_watchdog+=1
        
        if earlystopping_watchdog > earlystopping_tolerance:
            accelerator.print("Early stopping triggered")
            break
        
    
    model_to_save = accelerator.unwrap_model(model)
    accelerator.save(model_to_save.state_dict(), os.path.join(save_dir, "model_final.pth"))
    
    
    test_success = test_model(
        model, test_dataloader, tokenizer, device, save_dir, save_dir
    )
    
    return True
    
        
        


def main():
    
    params = argparse.ArgumentParser(description="Finetune model")

    params.add_argument(
        "--pretrained_model",
        required=False,
        default=None,
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
        required=True,
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
        "--lr",
        required=True,
        default=0.00002,
        type=float,
        help="Maximum learning rate.",
    )
    params.add_argument(
        "--epochs",
        required=False,
        default=50,
        type=int,
        help="The number of total training steps/batches.",
    )
    params.add_argument(
        "--patience",
        required=False,
        default=5,
        type=int,
        help="The number of epochs to wait before reducing learning rate.",
    )
    params.add_argument(
        "--train",
        required=True,
        type=str,
        help="The path of the training set file",
    )
    params.add_argument(
        "--val",
        required=False,
        type=str,
        help="The path of the validation set file",
    )
    params.add_argument(
        "--test",
        required=True,
        type=str,
        help="The path of the test set file",
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
        "--test_only",
        action="store_true",
        help="只运行测试模式"
    )
    params.add_argument(
        "--test_result_dir",
        required=False,
        type=str,
        help="The path of the test result directory",
    )
    params.add_argument(
        "--result_dir",
        required=True,
        default="./",
        type=str,
        help="The saving path of the finetuned model and/or evaluation result",
    )
    params.add_argument(
        "--kmer",
        required=True,
        type=int,
        help="K-mer",
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
    
    pretrained_model_path = args.pretrained_model
    tokenizer_path = args.tokenizer_path
    kmer = args.kmer

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    patience = args.patience
    train_file = args.train
    val_file = args.val
    test_file = args.test
    bw_file = args.bw
    reference_genome = args.reference_genome
    
    result_dir = args.result_dir
    test_result_dir = args.test_result_dir if args.test_only else result_dir

    
    if accelerator.is_main_process:
        accelerator.print("Model will be saved to:", result_dir)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(test_result_dir, exist_ok=True)
        
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    atac_model = AutoModel.from_pretrained(
        pretrained_model_path, add_pooling_layer=False
    )

    
    if args.test_only:
        
        if accelerator.is_main_process:
            accelerator.print("Running in test-only mode")
        
        test_dataset = PairDataset(test_file, bw_file, reference_genome, tokenizer)
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            prefetch_factor=2, 
            persistent_workers=True,
            num_workers=8,
            collate_fn=lambda x: collate_fn(x, tokenizer)
        )

        
        model = PairClassificationModel(atac_model)
        test_dataloader, model = accelerator.prepare(test_dataloader, model)

        
        test_success = test_model(
            model, test_dataloader, tokenizer, device, result_dir, test_result_dir
        )
        if not test_success:
            accelerator.print("模型测试失败")
    else:
        train_dataset = PairDataset(train_file, bw_file, reference_genome, tokenizer)
        val_dataset = PairDataset(val_file, bw_file, reference_genome, tokenizer)
        test_dataset = PairDataset(test_file, bw_file, reference_genome, tokenizer)

        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            prefetch_factor=2, 
            persistent_workers=True,
            num_workers=8,
            collate_fn=lambda x: collate_fn(x, tokenizer)
        )
        val_dataloader = DataLoader(
            dataset=val_dataset, 
            batch_size=batch_size, 
            prefetch_factor=2, 
            persistent_workers=True,
            num_workers=8,
            shuffle=False, 
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, tokenizer)
        )
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size, 
            prefetch_factor=2, 
            persistent_workers=True,
            num_workers=8,
            collate_fn=lambda x: collate_fn(x, tokenizer)
        )

        
        model = PairClassificationModel(atac_model)
        
        criterion = nn.BCELoss()
        optim = torch.optim.AdamW(
            model.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1
        )
        scheduler = ReduceLROnPlateau(
            optim, mode='max', factor=0.5, patience=2, verbose=True
        )
        train_dataloader, val_dataloader, test_dataloader, model, optim, scheduler = (
            accelerator.prepare(
                train_dataloader, val_dataloader, test_dataloader, model, optim, scheduler
            )
        )

        finetune_model(
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            tokenizer,
            criterion,
            optim,
            scheduler,
            epochs,
            patience,
            result_dir,
        )

        accelerator.end_training()
        
    
    if accelerator.is_main_process:
        
        params_dict = vars(args)
        with open(os.path.join(result_dir, 'training_params.json'), 'w') as f:
            json.dump(params_dict, f, indent=4)
    
    gc.collect()

    
    torch.cuda.empty_cache()
    
    return True

if __name__ == "__main__":
    main()
    
    
    
    os.abort()