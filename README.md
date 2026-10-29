<img width="7483" height="7700" alt="Tagger Preview Workflow" src="https://github.com/user-attachments/assets/a315d813-5fc4-409a-b42b-c079a5d362a4" />




# PixAI Tagger (upd to v1.0)

Official source:  https://huggingface.co/pixai-labs/pixai-tagger-v1.0

Download two files (any path supported):
```
model.safetensors
```

```
config.json
```

Many thanks to [PixAI Labs](https://huggingface.co/pixai-labs)

---

# AnimeTimm Tagger
*Support is implemented only for the convnextv2_huge.dbv4-full*

Official source: https://huggingface.co/animetimm/convnextv2_huge.dbv4-full

ONNX conversion: https://huggingface.co/itterative/convnextv2_huge.dbv4-full-onnx

Download two files (any path supported):

```
model.onnx
```
*"model.onnx data" file must be located in the same folder as "model.onnx"*

```
selected_tags.csv
```

Many thanks to [AnimeTimm](https://huggingface.co/animetimm)


---

# Camie Tagger

Official source: [https://huggingface.co/Camais03/camie-tagger-v2](https://huggingface.co/Camais03/camie-tagger-v2)

Download two files (any path supported):

```
camie-tagger-v2.onnx
```

```
camie-tagger-v2-metadata.json
```

Many thanks to [Camais03](https://huggingface.co/Camais03)

---


# Taggerine Tagger

Official source: https://huggingface.co/lodestones/taggerine

Download two files (any path supported):

```
tagger_proto.safetensors
```

```
tagger_vocab_with_categories_and_alias_updated.json
```

Many thanks to [lodestones](https://huggingface.co/lodestones)

---

# Tag Combiner

You can pass the results from the taggers into this node and combine them by using subtraction or by merging unique tags.
