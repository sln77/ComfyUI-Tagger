import os
import csv
import numpy as np
import torch
from PIL import Image
import onnxruntime as ort

class ConvNeXtV2TaggerNode:

    def __init__(self):
        self.loaded_model = None
        self.cached_onnx_path = ""
        self.cached_csv_path = ""
        self.tags = []
        self.tag_categories = {}
        self.model_img_size = 512

    CATEGORIES = ['character', 'general', 'rating']

    RETURN_TYPES = ("STRING",) + tuple("STRING" for _ in CATEGORIES)
    RETURN_NAMES = ("all_tags",) + tuple(cat + "_tags" for cat in CATEGORIES)

    FUNCTION = "tag_image"
    CATEGORY = "Image/Tagger"

    @classmethod
    def INPUT_TYPES(cls):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_onnx_path = os.path.join(base_dir, "models", "convnextv2_huge", "model.onnx")
        default_csv_path = os.path.join(base_dir, "models", "convnextv2_huge", "selected_tags.csv")

        return {
            "required": {
                "image": ("IMAGE",),
                "onnx_path": ("STRING", {"default": default_onnx_path}),
                "csv_path": ("STRING", {"default": default_csv_path}),
                "threshold": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "character_threshold": ("FLOAT", {
                    "default": 0.50,
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

    def load_model(self, onnx_path: str, csv_path: str):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tags CSV file not found at: {csv_path}")

        onnx_dir = os.path.dirname(onnx_path)
        expected_data_files = [
            onnx_path + "_data",
            os.path.join(onnx_dir, "model.onnx_data"),
            onnx_path + ".data",
            os.path.join(onnx_dir, "model.onnx.data"),
        ]
        if not any(os.path.exists(p) for p in expected_data_files):
            print(f"[ConvNeXtV2] WARNING: 'model.onnx_data' not found in {onnx_dir}. "
                  f"Large ONNX models require external tensor weights file in the same directory.")

        available_providers = ort.get_available_providers()
        providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]
        if not providers:
            providers = ['CPUExecutionProvider']

        self.loaded_model = ort.InferenceSession(onnx_path, providers=providers)

        try:
            shape = self.loaded_model.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
                self.model_img_size = shape[2]
            else:
                self.model_img_size = 512
        except Exception:
            self.model_img_size = 512

        self.tags = []
        self.tag_categories = {}

        category_map = {
            "0": "general",
            "4": "character",
            "9": "rating",
            "1": "artist",
            "3": "copyright",
            "5": "meta",
        }

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            name_idx = 1
            cat_idx = 2

            if header:
                header_lower = [h.strip().lower() for h in header]
                if "name" in header_lower:
                    name_idx = header_lower.index("name")
                elif "tag" in header_lower:
                    name_idx = header_lower.index("tag")

                if "category" in header_lower:
                    cat_idx = header_lower.index("category")

            for row in reader:
                if not row or len(row) <= max(name_idx, cat_idx):
                    continue
                tag_name = row[name_idx].strip()
                raw_cat = row[cat_idx].strip().lower()

                cat = category_map.get(raw_cat, raw_cat)
                if cat not in self.CATEGORIES:
                    cat = "general"

                self.tags.append(tag_name)
                self.tag_categories[tag_name] = cat

        self.cached_onnx_path = onnx_path
        self.cached_csv_path = csv_path

    def tag_image(self, image: torch.Tensor, onnx_path: str, csv_path: str, 
                threshold: float, character_threshold: float, 
                replace_underscores: bool, exclude_tags: str):

        if (self.loaded_model is None or 
            self.cached_onnx_path != onnx_path or 
            self.cached_csv_path != csv_path):
            self.load_model(onnx_path, csv_path)

        input_name = self.loaded_model.get_inputs()[0].name
        output_name = self.loaded_model.get_outputs()[0].name

        excluded = set()
        for tag in exclude_tags.split(','):
            tag = tag.strip().lower()
            if not tag:
                continue
            if replace_underscores:
                tag = tag.replace(' ', '_')
            excluded.add(tag)

        batch_size = image.shape[0]
        batch_all_tags = []
        batch_cat_tags = {cat: [] for cat in self.CATEGORIES}

        for b in range(batch_size):
            img_pil = self.tensor_to_pil(image[b])
            processed_image = self.preprocess(img_pil, self.model_img_size)

            logits = self.loaded_model.run([output_name], {input_name: processed_image})[0][0]

            logits = np.clip(logits, -50.0, 50.0)
            probs = 1.0 / (1.0 + np.exp(-logits))

            tags_by_category = {cat: [] for cat in self.CATEGORIES}

            for i, prob in enumerate(probs):
                tag_name = self.tags[i]
                category = self.tag_categories.get(tag_name, "general")

                curr_thresh = character_threshold if category == "character" else threshold
                if prob < curr_thresh:
                    continue

                if tag_name.lower() in excluded:
                    continue

                tags_by_category[category].append((tag_name, float(prob)))

            single_output_tags = {}
            all_tags_list = []

            for category in self.CATEGORIES:
                sorted_tags = sorted(tags_by_category.get(category, []), key=lambda x: x[1], reverse=True)

                formatted_tags = []
                for tag, prob in sorted_tags:
                    display_tag = tag.replace('_', ' ') if replace_underscores else tag
                    formatted_tags.append(display_tag)

                cat_str = ", ".join(formatted_tags)
                single_output_tags[category] = cat_str

                if category in ('character', 'general'):
                    all_tags_list.extend(formatted_tags)

            batch_all_tags.append(", ".join(all_tags_list))
            for cat in self.CATEGORIES:
                batch_cat_tags[cat].append(single_output_tags[cat])

        final_output = ["\n".join(batch_all_tags)]
        for category in self.CATEGORIES:
            final_output.append("\n".join(batch_cat_tags[category]))

        return tuple(final_output)

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        image_np = tensor.cpu().numpy()
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def preprocess(self, image: Image.Image, img_size: int) -> np.ndarray:
        w, h = image.size
        scale = float(img_size) / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized = image.resize((new_w, new_h), Image.BICUBIC)
        square = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        square.paste(resized, ((img_size - new_w) // 2, (img_size - new_h) // 2))

        image_np = np.array(square).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_np = (image_np - mean) / std

        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)
        return image_np