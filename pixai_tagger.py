import os
import json
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
import timm
from PIL import Image

class TaggingHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head = nn.Sequential(nn.Linear(input_dim, num_classes))

    def forward(self, x):
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return probs

def get_model():
    base_model_repo = "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"
    encoder = timm.create_model(base_model_repo, pretrained=False)
    encoder.reset_classifier(0)
    
    decoder = TaggingHead(1024, 13461)
    model = nn.Sequential(encoder, decoder)
    return model

class PixAITagger:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_files = {"model": "", "tags": "", "map": ""}
        
        self.tag_map = {}
        self.index_to_tag_map = {}
        self.character_ip_mapping = {}
        self.gen_tag_count = 0
        self.character_tag_count = 0

    @classmethod
    def INPUT_TYPES(s):
        base_path = r""
        
        return {
            "required": {
                "image": ("IMAGE", ),
                "model_file": ("STRING", {"default": os.path.join(base_path, "model_v0.9.pth"), "multiline": False}),
                "tags_file": ("STRING", {"default": os.path.join(base_path, "tags_v0.9_13k.json"), "multiline": False}),
                "char_map_file": ("STRING", {"default": os.path.join(base_path, "char_ip_map.json"), "multiline": False}),
                
                "general_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": True, "placeholder": "exclude_tags"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("tags_string", "character_tags", "ip_tags", "general_tags")
    FUNCTION = "tag_image"
    CATEGORY = "Image/Tagger"

    def load_resources(self, model_file, tags_file, map_file):
        model_file = os.path.abspath(model_file)
        tags_file = os.path.abspath(tags_file)
        map_file = os.path.abspath(map_file)

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"PixAI Model file not found: {model_file}")
        if not os.path.exists(tags_file):
            raise FileNotFoundError(f"Tags JSON file not found: {tags_file}")
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Mapping JSON file not found: {map_file}")

        files_changed = (tags_file != self.loaded_files["tags"]) or (map_file != self.loaded_files["map"])
        
        if files_changed:
            print(f"[PixAI Tagger] Loading tags/maps from {tags_file}...")
            with open(tags_file, "r", encoding="utf-8") as f:
                tag_info = json.load(f)
                self.tag_map = tag_info["tag_map"]
                tag_split = tag_info["tag_split"]
                self.gen_tag_count = tag_split["gen_tag_count"]
                self.character_tag_count = tag_split["character_tag_count"]
            
            self.index_to_tag_map = {v: k for k, v in self.tag_map.items()}

            with open(map_file, "r", encoding="utf-8") as f:
                self.character_ip_mapping = json.load(f)
            
            self.loaded_files["tags"] = tags_file
            self.loaded_files["map"] = map_file

        if self.model is None or model_file != self.loaded_files["model"]:
            print(f"[PixAI Tagger] Loading model weights from {model_file}...")
            if self.model is None:
                self.model = get_model()
                self.model.to(self.device)

            try:
                states_dict = torch.load(model_file, map_location=self.device, weights_only=True)
            except TypeError:
                states_dict = torch.load(model_file, map_location=self.device)
                
            self.model.load_state_dict(states_dict)
            self.model.eval()
            self.loaded_files["model"] = model_file
            print("[PixAI Tagger] Model loaded successfully.")

    def tag_image(self, image, model_file, tags_file, char_map_file, general_threshold, character_threshold, replace_underscore, exclude_tags):
        self.load_resources(model_file, tags_file, char_map_file)
        
        exclusions = set([x.strip() for x in exclude_tags.split(",") if x.strip()])
        
        final_tags_list = []
        char_tags_list = []
        ip_tags_list = []
        gen_tags_list = []

        def process_tags(tags):
            processed = []
            for t in tags:
                if replace_underscore:
                    t = t.replace("_", " ")
                
                if t not in exclusions:
                    processed.append(t)
            return processed

        for i in range(image.shape[0]):
            img_tensor = image[i] 
            img_tensor = img_tensor.permute(2, 0, 1) 
            img_tensor = TF.resize(img_tensor, (448, 448), interpolation=transforms.InterpolationMode.BICUBIC)
            img_tensor = TF.normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.inference_mode():
                probs = self.model(img_tensor)[0]

            general_mask = probs[:self.gen_tag_count] > general_threshold
            character_mask = probs[self.gen_tag_count:] > character_threshold
            
            general_indices = general_mask.nonzero(as_tuple=True)[0].cpu().tolist()
            character_indices = (character_mask.nonzero(as_tuple=True)[0] + self.gen_tag_count).cpu().tolist()
            
            raw_gen_tags = [self.index_to_tag_map[idx] for idx in general_indices]
            raw_char_tags = [self.index_to_tag_map[idx] for idx in character_indices]
            
            raw_ip_tags = set()
            if raw_char_tags:
                for ctag in raw_char_tags:
                    if ctag in self.character_ip_mapping:
                        for ip in self.character_ip_mapping[ctag]:
                            raw_ip_tags.add(ip)
            raw_ip_tags = sorted(list(raw_ip_tags))

            final_gen = process_tags(raw_gen_tags)
            final_char = process_tags(raw_char_tags)
            final_ip = process_tags(raw_ip_tags)

            all_tags_combined = final_gen + final_char + final_ip
            
            final_tags_list.append(", ".join(all_tags_combined))
            char_tags_list.append(", ".join(final_char))
            ip_tags_list.append(", ".join(final_ip))
            gen_tags_list.append(", ".join(final_gen))

        return (final_tags_list, char_tags_list, ip_tags_list, gen_tags_list)