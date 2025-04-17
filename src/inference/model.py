import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, IA3Config, TaskType
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import os
import json
import glob

class PairClassificationModel(
    nn.Module,
    PyTorchModelHubMixin
):
    def __init__(self, base_model=None):
        super(PairClassificationModel, self).__init__()
        
        if base_model is None or isinstance(base_model, str):
            model_id = base_model if isinstance(base_model, str) else "chikit2077/CLAMP-6mer-1500bp-pretrain"
            base_model = AutoModel.from_pretrained(model_id, add_pooling_layer=False)

        peft_config = IA3Config(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            target_modules=["key", "value", "output.dense"],
            feedforward_modules=["output.dense"],
        )
        self.base_model = get_peft_model(base_model, peft_config)
        
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

    @classmethod
    def _from_pretrained(cls, model_id, **kwargs):
        """
        Custom from_pretrained method that supports loading models from local or Hub, prioritizing pytorch_model.bin
        """
        print(f"Loading model: {model_id}")
        
        
        is_local = os.path.isdir(model_id)
        
        if is_local:
            
            print(f"Loading from local path: {model_id}")
            
            
            config_path = os.path.join(model_id, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                print("Configuration file loaded successfully")
            else:
                config = {}
                print("Configuration file not found, using default settings")
            
            
            base_model = kwargs.get("base_model", config.get("base_model", "chikit2077/CLAMP-6mer-1500bp-pretrain"))
            
            
            model = cls(base_model=base_model)
            
            
            pytorch_model_path = os.path.join(model_id, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                print(f"Loading PyTorch weights: {pytorch_model_path}")
                try:
                    model.load_state_dict(torch.load(pytorch_model_path, weights_only=True))
                    print("PyTorch weights loaded successfully")
                    return model
                except Exception as e:
                    print(f"PyTorch weights loading failed: {e}")
            
        else:
            
            print(f"Loading from Hugging Face Hub: {model_id}")
            try:
                
                try:
                    model_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
                    print(f"Successfully downloaded PyTorch weights: {model_file}")
                    
                    
                    try:
                        config_file = hf_hub_download(repo_id=model_id, filename="config.json")
                        with open(config_file, "r") as f:
                            config = json.load(f)
                    except:
                        config = {}
                        print("Unable to download config file, using default settings")
                    
                    
                    base_model = kwargs.get("base_model", config.get("base_model", "chikit2077/CLAMP-6mer-1500bp-pretrain"))
                    model = cls(base_model=base_model)
                    
                    
                    model.load_state_dict(torch.load(model_file, weights_only=True))
                    print("Model loaded successfully")
                    return model
                    
                except Exception as e:
                    print(f"Custom loading method failed: {e}")
                    print("Falling back to standard PyTorchModelHubMixin.from_pretrained method")
            
            except Exception as hub_error:
                print(f"Loading from Hub failed: {hub_error}")
            
            
            return super().from_pretrained(model_id, **kwargs)
    
    
    def _save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        
        config = {
            "architecture": {
                "signal_input_dim": 64,
                "signal_hidden_dim": 128,
                "fusion_hidden_dim": 512,
                "attention_heads": 8,
                "dropout": 0.1,
                "distance_hidden_dim": 32,
                "distance_output_dim": 64,
                "classifier_hidden_dim": 256
            }
        }
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        return [model_path]