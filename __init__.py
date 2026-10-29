from .camie_tagger import CamieTaggerNode
from .pixai_tagger import PixAITagger
from .taggerine_tagger import TaggerineTaggerNode
from .tag_combiner import TagCombiner
from .convnextv2_tagger import ConvNeXtV2TaggerNode

NODE_CLASS_MAPPINGS = {
    "CamieTaggerNode": CamieTaggerNode,
    "PixAITagger": PixAITagger,
    "TaggerineTaggerNode": TaggerineTaggerNode,
    "TagCombiner": TagCombiner,
    "ConvNeXtV2TaggerNode": ConvNeXtV2TaggerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamieTaggerNode": "Camie Tagger",
    "PixAITagger": "PixAI Tagger",
    "TaggerineTaggerNode": "Taggerine Tagger",
    "ConvNeXtV2TaggerNode": "AnimeTimm Tagger",
    "TagCombiner": "Tag Combiner"

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ Tagger nodes loaded successfully.")
