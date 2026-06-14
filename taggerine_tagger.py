
import os
import json
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file


D_MODEL = 1280
N_HEADS = 20
HEAD_DIM = D_MODEL // N_HEADS  # 64
N_LAYERS = 32
D_FFN = 5120
N_REGISTERS = 4
PATCH_SIZE = 16
ROPE_THETA = 100.0
ROPE_RESCALE = 2.0
LN_EPS = 1e-5
LAYERSCALE = 1.0

FEATURE_DIM = (1 + N_REGISTERS) * D_MODEL  # 6400

CATEGORY_NAMES = {
    0: 'general',
    1: 'artist',
    2: 'colorist',
    3: 'copyright',
    4: 'character',
    5: 'species',
    6: 'meta',
    7: 'style',
    8: 'lore'
}


@lru_cache(maxsize=32)
def _patch_coords_cached(h: int, w: int, device_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    cy = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
    cx = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
    coords = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    coords = coords * ROPE_RESCALE
    return coords


def _build_rope(h_patches: int, w_patches: int, dtype: torch.dtype, device: torch.device):
    coords = _patch_coords_cached(h_patches, w_patches, str(device))
    inv_freq = 1.0 / (ROPE_THETA ** torch.arange(0, 1, 4 / HEAD_DIM, dtype=torch.float32, device=device))
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    n_pre = 1 + N_REGISTERS
    q_pre, q_pat = q[..., :n_pre, :], q[..., n_pre:, :]
    k_pre, k_pat = k[..., :n_pre, :], k[..., n_pre:, :]
    q_pat = q_pat * cos + _rotate_half(q_pat) * sin
    k_pat = k_pat * cos + _rotate_half(k_pat) * sin
    return torch.cat([q_pre, q_pat], dim=-2), torch.cat([k_pre, k_pat], dim=-2)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, D_MODEL))


class _GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.up_proj = nn.Linear(D_MODEL, D_FFN, bias=True)
        self.down_proj = nn.Linear(D_FFN, D_MODEL, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.attention = _Attention()
        self.layer_scale1 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))
        self.norm2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.mlp = _GatedMLP()
        self.layer_scale2 = nn.Parameter(torch.full((D_MODEL,), LAYERSCALE))

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin) * self.layer_scale1
        x = x + self.mlp(self.norm2(x)) * self.layer_scale2
        return x


class _Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.register_tokens = nn.Parameter(torch.zeros(1, N_REGISTERS, D_MODEL))
        self.patch_embeddings = nn.Conv2d(3, D_MODEL, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        dtype = self.patch_embeddings.weight.dtype
        patches = self.patch_embeddings(pixel_values.to(dtype)).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        return torch.cat([cls, regs, patches], dim=1)


class DINOv3ViTH(nn.Module):
    """DINOv3 ViT-H/16+ backbone."""
    
    def __init__(self):
        super().__init__()
        self.embeddings = _Embeddings()
        self.layer = nn.ModuleList([_Block() for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL, eps=LN_EPS)

    def forward(self, pixel_values):
        _, _, H, W = pixel_values.shape
        x = self.embeddings(pixel_values)
        h_p, w_p = H // PATCH_SIZE, W // PATCH_SIZE
        cos, sin = _build_rope(h_p, w_p, x.dtype, pixel_values.device)
        for block in self.layer:
            x = block(x, cos, sin)
        return self.norm(x)


class _LowRankHead(nn.Module):
    """Two-matrix low-rank projection head."""
    
    def __init__(self, in_dim: int, rank: int, num_tags: int, down_bias: bool, up_bias: bool):
        super().__init__()
        self.proj_down = nn.Linear(in_dim, rank, bias=down_bias)
        self.proj_up = nn.Linear(rank, num_tags, bias=up_bias)

    def forward(self, x):
        return self.proj_up(self.proj_down(x))


def _build_head_from_checkpoint(head_sd: dict, in_dim: int, num_tags: int):
    """Build head from checkpoint state dict."""
    weights_2d = [(k, v) for k, v in head_sd.items() if k.endswith(".weight") and v.ndim == 2]

    # Case 1: single dense linear
    singles = [(k, v) for k, v in weights_2d if tuple(v.shape) == (num_tags, in_dim)]
    if len(weights_2d) <= 2 and len(singles) == 1:
        wkey, wval = singles[0]
        base = wkey[:-len(".weight")]
        bias_key = base + ".bias"
        has_bias = bias_key in head_sd
        module = nn.Linear(in_dim, num_tags, bias=has_bias)
        remapped = {"weight": wval}
        if has_bias:
            remapped["bias"] = head_sd[bias_key]
        return module, remapped

    down = None
    up = None
    for k, v in weights_2d:
        if v.shape[1] == in_dim and v.shape[0] != num_tags:
            down = (k, v)
        elif v.shape[0] == num_tags and v.shape[1] != in_dim:
            up = (k, v)

    if down is not None and up is not None:
        rank_down = down[1].shape[0]
        rank_up = up[1].shape[1]
        if rank_down != rank_up:
            raise RuntimeError(f"Low-rank head: inner dims disagree (down out={rank_down}, up in={rank_up})")

        down_key, down_w = down
        up_key, up_w = up
        down_base = down_key[:-len(".weight")]
        up_base = up_key[:-len(".weight")]
        down_bias_key = down_base + ".bias"
        up_bias_key = up_base + ".bias"
        has_down_bias = down_bias_key in head_sd
        has_up_bias = up_bias_key in head_sd

        module = _LowRankHead(in_dim, rank_down, num_tags, has_down_bias, has_up_bias)
        remapped = {"proj_down.weight": down_w, "proj_up.weight": up_w}
        if has_down_bias:
            remapped["proj_down.bias"] = head_sd[down_bias_key]
        if has_up_bias:
            remapped["proj_up.bias"] = head_sd[up_bias_key]

        print(f"[Taggerine] Low-rank head: in_dim={in_dim}, rank={rank_down}, num_tags={num_tags}")
        return module, remapped

    raise RuntimeError("Could not infer head architecture from checkpoint.")


class DINOv3Tagger(nn.Module):
    """Backbone + head."""
    
    def __init__(self):
        super().__init__()
        self.backbone = DINOv3ViTH()
        self.head = None

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values)
        cls = hidden[:, 0, :]
        regs = hidden[:, 1: 1 + N_REGISTERS, :].flatten(1)
        features = torch.cat([cls, regs], dim=-1).float()
        return self.head(features)


def _split_and_clean_state_dict(sd: dict):
    """Split state dict into backbone and head."""
    backbone_sd = {}
    head_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            nk = k[len("backbone."):]
            if nk.startswith("model.layer."):
                nk = nk[len("model."):]
            backbone_sd[nk] = v
        else:
            head_sd[k] = v

    # Remap layer_scale
    for k in list(backbone_sd.keys()):
        if ".layer_scale" in k and k.endswith(".lambda1"):
            backbone_sd[k[:-len(".lambda1")]] = backbone_sd.pop(k)

    # Drop rope buffers
    for k in list(backbone_sd.keys()):
        if "rope_embeddings" in k:
            backbone_sd.pop(k)

    return backbone_sd, head_sd



class TaggerineTaggerNode:
    
    def __init__(self):
        self.model = None
        self.vocab = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cached_model_path = ""
        self.cached_vocab_path = ""

    CATEGORIES = sorted(['general', 'artist', 'copyright', 'character', 'species', 'meta', 'style', 'lore'])
    
    RETURN_TYPES = ("STRING",) + tuple("STRING" for _ in CATEGORIES)
    RETURN_NAMES = ("all_tags",) + tuple(cat + "_tags" for cat in CATEGORIES)

    FUNCTION = "tag_image"
    CATEGORY = "Image/Tagger"

    @classmethod
    def INPUT_TYPES(cls):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(base_dir, "models", "tagger_proto.safetensors")
        default_vocab_path = os.path.join(base_dir, "models", "tagger_vocab_with_categories_and_alias_updated.json")

        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": default_model_path}),
                "vocab_path": ("STRING", {"default": default_vocab_path}),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_size": ("INT", {
                    "default": 512,
                    "min": 224,
                    "max": 1024,
                    "step": 16
                }),
                "replace_underscores": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "exclude_tags": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def load_model(self, model_path, vocab_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")

        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = {
            'idx2tag': vocab_data.get('idx2tag', []),
            'tag2category': vocab_data.get('tag2category', {})
        }
        
        num_tags = len(self.vocab['idx2tag'])
        
        print(f"[Taggerine] Loading checkpoint: {model_path}")
        print(f"[Taggerine] Vocabulary: {num_tags:,} tags")
        
        sd = load_file(model_path, device="cpu")
        backbone_sd, head_sd = _split_and_clean_state_dict(sd)
        
        if not head_sd:
            raise RuntimeError("Checkpoint contains no head weights")
        
        self.model = DINOv3Tagger()
        head_module, head_sd_remapped = _build_head_from_checkpoint(head_sd, FEATURE_DIM, num_tags)
        self.model.head = head_module
        
        self.model.backbone.load_state_dict(backbone_sd, strict=True)
        self.model.head.load_state_dict(head_sd_remapped, strict=True)
        
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.model.backbone = self.model.backbone.to(device=self.device, dtype=dtype)
        self.model.head = self.model.head.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        
        self.cached_model_path = model_path
        self.cached_vocab_path = vocab_path
        
        print(f"[Taggerine] Model loaded successfully on {self.device}")

    def tag_image(self, image: torch.Tensor, model_path: str, vocab_path: str, 
                  threshold: float, max_size: int, replace_underscores: bool, 
                  exclude_tags: str = ""):
        
        if self.model is None or self.cached_model_path != model_path or self.cached_vocab_path != vocab_path:
            self.load_model(model_path, vocab_path)

        img_pil = self.tensor_to_pil(image)
        
        processed = self.preprocess(img_pil, max_size).to(self.device)

        with torch.inference_mode():
            logits = self.model(processed)[0]
            probs = torch.sigmoid(logits.float()).cpu().numpy()

        idx2tag = self.vocab['idx2tag']
        tag2category = self.vocab['tag2category']
        
        excluded = set()
        for tag in exclude_tags.split(','):
            tag = tag.strip().lower()
            if tag:
                excluded.add(tag.replace(' ', '_') if not replace_underscores else tag)

        tags_by_category = {cat: [] for cat in self.CATEGORIES}
        
        for i, prob in enumerate(probs):
            if prob > threshold and i < len(idx2tag):
                tag_name = idx2tag[i]
                
                if tag_name.lower() in excluded:
                    continue
                
                cat_num = tag2category.get(tag_name, 0)
                category = CATEGORY_NAMES.get(cat_num, 'general')
                
                if category == 'colorist':
                    category = 'general'
                
                if category in tags_by_category:
                    tags_by_category[category].append((tag_name, prob))
                else:
                    tags_by_category['general'].append((tag_name, prob))

        output_tags = {}
        all_tags_list = []

        for category in self.CATEGORIES:
            sorted_tags = sorted(tags_by_category[category], key=lambda x: x[1], reverse=True)
            
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
        """Convert ComfyUI tensor to PIL Image"""
        image_np = tensor.squeeze().cpu().numpy()
        image_np = (image_np * 255).astype('uint8')
        return Image.fromarray(image_np, 'RGB')

    def preprocess(self, image: Image.Image, max_size: int) -> torch.Tensor:
        """Preprocess image for Taggerine model"""
        w, h = image.size
        
        long_edge = max(w, h)
        target_long = max(PATCH_SIZE, (min(long_edge, max_size) // PATCH_SIZE) * PATCH_SIZE)
        scale = target_long / long_edge
        
        new_w = max(PATCH_SIZE, (round(w * scale) // PATCH_SIZE) * PATCH_SIZE)
        new_h = max(PATCH_SIZE, (round(h * scale) // PATCH_SIZE) * PATCH_SIZE)
        
        transform = transforms.Compose([
            transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
