from .camie_tagger import CamieTaggerNode
from .pixai_tagger import PixAITagger
from .taggerine_tagger import TaggerineTaggerNode
from .tag_combiner import TagCombiner

NODE_CLASS_MAPPINGS = {
    "CamieTaggerNode": CamieTaggerNode,
    "PixAITagger": PixAITagger,
    "TaggerineTaggerNode": TaggerineTaggerNode,
    "TagCombiner": TagCombiner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamieTaggerNode": "Camie Tagger",
    "PixAITagger": "PixAI Tagger",
    "TaggerineTaggerNode": "Taggerine Tagger",
    "TagCombiner": "Tag Combiner"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ Tagger nodes loaded successfully.")
