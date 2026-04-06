import os
import json
import onnxruntime as ort
import numpy as np
import torch
from PIL import Image

class CamieTaggerNode:
    
    def __init__(self):
        self.loaded_model = None
        self.loaded_metadata = None
        self.cached_onnx_path = ""
        self.cached_metadata_path = ""

    CATEGORIES = sorted(['character', 'general', 'meta', 'copyright', 'artist', 'year', 'rating'])
    
    RETURN_TYPES = ("STRING",) + tuple("STRING" for _ in CATEGORIES)
    RETURN_NAMES = ("all_tags",) + tuple(cat + "_tags" for cat in CATEGORIES)

    FUNCTION = "tag_image"
    CATEGORY = "Image/Tagger"

    @classmethod
    def INPUT_TYPES(cls):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_onnx_path = os.path.join(base_dir, "models", "camie-tagger-v2.onnx")
        default_metadata_path = os.path.join(base_dir, "models", "camie-tagger-v2-metadata.json")

        return {
            "required": {
                "image": ("IMAGE",),
                "onnx_path": ("STRING", {"default": default_onnx_path}),
                "metadata_path": ("STRING", {"default": default_metadata_path}),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "replace_underscores": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "exclude_tags": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def load_model(self, onnx_path, metadata_path):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata JSON file not found at: {metadata_path}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.loaded_model = ort.InferenceSession(onnx_path, providers=providers)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.loaded_metadata = json.load(f)
            
        self.cached_onnx_path = onnx_path
        self.cached_metadata_path = metadata_path

    def tag_image(self, image: torch.Tensor, onnx_path: str, metadata_path: str, threshold: float, replace_underscores: bool, exclude_tags: str):
        if self.loaded_model is None or self.cached_onnx_path != onnx_path or self.cached_metadata_path != metadata_path:
            self.load_model(onnx_path, metadata_path)

        img_pil = self.tensor_to_pil(image)
        img_size = self.loaded_metadata['model_info']['img_size']
        processed_image = self.preprocess(img_pil, img_size)

        input_name = self.loaded_model.get_inputs()[0].name
        output_name = self.loaded_model.get_outputs()[0].name
        
        logits = self.loaded_model.run([output_name], {input_name: processed_image})[0][0]
        probs = 1 / (1 + np.exp(-logits)) 

        dataset_info = self.loaded_metadata['dataset_info']
        tag_mapping = dataset_info['tag_mapping']
        
        if 'idx_to_tag' in tag_mapping:
            idx_map = {int(k): v for k, v in tag_mapping['idx_to_tag'].items()}
            tags = [idx_map[i] for i in range(len(idx_map))]
        else:
            tags = [tag for tag, idx in sorted(tag_mapping['tag_to_idx'].items(), key=lambda item: item[1])]

        tag_to_category = tag_mapping.get('tag_to_category', {})
        
        excluded = set()
        for tag in exclude_tags.split(','):
            tag = tag.strip().lower()
            if not tag:
                continue
            if replace_underscores:
                tag = tag.replace(' ', '_')
            excluded.add(tag)

        tags_by_category = {cat: [] for cat in self.CATEGORIES}
        
        for i, prob in enumerate(probs):
            if prob > threshold:
                tag_name = tags[i]
                
                if tag_name.lower() in excluded:
                    continue
                
                category = tag_to_category.get(tag_name, "general")
                
                if category in tags_by_category:
                    tags_by_category[category].append((tag_name, prob))
                else:
                    tags_by_category["general"].append((tag_name, prob))

        output_tags = {}
        all_tags_list = []

        for category in self.CATEGORIES:
            sorted_tags = sorted(tags_by_category.get(category, []), key=lambda x: x[1], reverse=True)
            
            formatted_tags = []
            for tag, prob in sorted_tags:
                display_tag = tag.replace('_', ' ') if replace_underscores else tag
                formatted_tags.append(display_tag)
            
            output_tags[category + "_tags"] = ", ".join(formatted_tags)
            all_tags_list.extend(formatted_tags)
            
        all_tags_str = ", ".join(all_tags_list)

        final_output = [all_tags_str]
        for category in self.CATEGORIES:
            final_output.append(output_tags[category + "_tags"])
            
        return tuple(final_output)

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        image_np = tensor.squeeze().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def preprocess(self, image: Image.Image, img_size: int) -> np.ndarray:
        image = image.resize((img_size, img_size), Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
        std = np.array([0.229, 0.224, 0.225]).astype(np.float32)
        
        image_np = (image_np - mean) / std
        
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)
        return image_np