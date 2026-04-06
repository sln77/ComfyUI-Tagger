from .camie_tagger import CamieTaggerNode
from .pixai_tagger import PixAITagger
from .tag_combiner import TagCombiner

NODE_CLASS_MAPPINGS = {
    "CamieTaggerNode": CamieTaggerNode,
    "PixAITagger": PixAITagger,
    "TagCombiner": TagCombiner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamieTaggerNode": "Camie Tagger",
    "PixAITagger": "PixAI Tagger",
    "TagCombiner": "Tag Combiner"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ Tagger nodes loaded successfully.")