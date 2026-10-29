import collections
from comfy_api.latest import io

class TagCombiner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.String.Input("text", default=""),
            prefix="text_",
            min=2,
            max=50
        )

        return io.Schema(
            node_id="TagCombiner",
            category="Image/Tagger",
            inputs=[
                io.Boolean.Input(
                    "only_unique_diff",
                    default=False,
                    label_on="Only Unique",
                    label_off="All unique"
                ),
                io.Autogrow.Input("texts", template=autogrow_template)
            ],
            outputs=[
                io.String.Output(display_name="combined_text")
            ]
        )

    @classmethod
    def execute(cls, only_unique_diff: bool, texts: io.Autogrow.Type) -> io.NodeOutput:
        sorted_keys = sorted(
            texts.keys(), 
            key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x
        )
        inputs = [texts[k] for k in sorted_keys]

        valid_inputs = [t for t in inputs if t is not None and isinstance(t, str) and t.strip() != ""]

        if not valid_inputs:
            return io.NodeOutput("")

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

        return io.NodeOutput(result_string)
