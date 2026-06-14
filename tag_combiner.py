import collections

class TagCombiner:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "only_unique_diff": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Only Unique", 
                    "label_off": "All unique"
                }),
            },
            "optional": {
                "text_1": ("STRING", {"forceInput": True}),
                "text_2": ("STRING", {"forceInput": True}),
                "text_3": ("STRING", {"forceInput": True}),
                "text_4": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)
    FUNCTION = "combine_tags"
    CATEGORY = "Image/Tagger"

    def combine_tags(self, only_unique_diff, text_1=None, text_2=None, text_3=None, text_4=None):
        inputs = [text_1, text_2, text_3, text_4,]
        valid_inputs = [t for t in inputs if t is not None and isinstance(t, str) and t.strip() != ""]

        if not valid_inputs:
            return ("",)

        all_tags_counter = collections.Counter()
        ordered_tags = []

        for text in valid_inputs:
            clean_text = text.replace('\n', ',')
            tags = [t.strip() for t in clean_text.split(',')]
            
            for tag in tags:
                if not tag: 
                    continue
                
                if tag not in all_tags_counter:
                    ordered_tags.append(tag)
                
                all_tags_counter[tag] += 1

        final_tags = []

        if only_unique_diff:
            for tag in ordered_tags:
                if all_tags_counter[tag] == 1:
                    final_tags.append(tag)
        else:
            final_tags = ordered_tags

        result_string = ", ".join(final_tags)

        return (result_string,)
