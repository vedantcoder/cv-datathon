# VQA Datathon — BITS F459 (student package)

Build a **small, fast** visual question answering model: it should be accurate and small and fast. Your model is run on a
**hidden test set on Codabench** ranked by a combined metric that rewards higher accuracy but penalizes higher parameter
count and FLOPs, so just accuracy is not enough, you have to make it efficient too!

## Contents

| Path | What |
|---|---|
| `data/processed/train/` | 15,000 images (224×224 JPEG), `questions.json` (79,676 q), `annotations.json` (10 human answers each) |
| `data/processed/val/` | 2,000 images, 10,388 questions + annotations (use for model selection) |
| `data/processed/answers.txt` | top-1000 normalized train answers (covers 86.5% of val majority answers) |
| `baseline/simple/` | baseline `model.py` + `words.txt` + `answers.txt` |
| `baseline/train_simple.py` | trains the baseline (~10 min on a Colab T4) |
| `data/checkpoints/simple_baseline.pt` | trained baseline checkpoint |
| `eval/vqa_eval.py` | official VQA accuracy (matches the official code exactly) |
| `eval/complexity.py` | params + FLOPs, exactly as Codabench measures them |
| `eval/check_submission.py` | builds `submission.zip` and runs **the Codabench ingestion program** on val |
| `codabench/ingestion_program/` | the program Codabench runs on your submission |
| `notebooks/student_baseline.ipynb` | Colab: train + evaluate the baseline |
| `notebooks/make_submission.ipynb` | Colab: build + check your submission zip |

Test images and questions are **not** included. Test runs only on Codabench.

## The interface (fixed)

```python
model = build_model()                                   # defined in your model.py
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu", weights_only=True))
answers = model.predict(images, questions)
```

- `images`: `FloatTensor[B, 3, 224, 224]`, **RGB, values in [0, 1], not mean/std-normalized** (PIL → RGB → `torchvision.transforms.functional.to_tensor`). Normalize inside your model.
- `questions`: `list[str]` (raw text, so use any tokenizer you like inside your model).
- return: `list[str]` of answers, one per question. **Use lowercase.**
- `build_model()` must work **offline**: all weights come from `checkpoint.pt`. Ship vocabularies and other small files next to `model.py` and read them relative to `__file__` (see the baseline).

## Scoring

**Accuracy** is the official VQA accuracy, in percent. Each answer is normalized (punctuation, number words → digits, articles, contractions) and scored against 10 human answers: the average over the 10 leave-one-out subsets of `min(#matches/3, 1)`. Quirk of the official metric: if all 10 humans agree, the match is exact and case-sensitive, so output lowercase.

**Params** = `sum(p.numel() for p in model.parameters())`. All parameters count, frozen pretrained backbones included.

**FLOPs** = `torch.utils.flop_counter.FlopCounterMode` around **one** `predict()` call with **batch size 1**, a fixed dummy image (all 0.5) and a fixed dummy question, on CPU, torch **2.14.1**. **One multiply-add = 2 FLOPs.** Matmul / conv / attention ops are counted; elementwise ops are not. Counts depend on the torch version, so use torch 2.14.1.

**Composite score** (ranking):

```
Score = Acc − 2·log2(Params / 18,154,866) − 2·log2(FLOPs / 3,688,267,776)
```

The reference values are the baseline's (val 46.16 / test 46.24 accuracy, so it scores 46.24). Each doubling of params costs 2 points and each doubling of FLOPs costs 2 points; halving either gains 2.

## Submitting on Codabench

1. Zip `model.py` + `checkpoint.pt` + helper files. Use `notebooks/make_submission.ipynb` or
   `python eval/check_submission.py --model-dir <dir> --ckpt <ckpt.pt> --out submission.zip`.
   It runs the **same** program Codabench runs, on val, on 2 CPU threads, and prints errors, val score, params, FLOPs and estimated test runtime.
2. Upload the zip on Codabench. It runs **on CPU** in a Docker image with python 3.10, torch 2.14.1, torchvision 0.29.1, transformers 5.20.2, pillow 12.3.0, numpy 2.2.6, pyyaml. There are **no other packages and no internet**.
3. The leaderboard shows the **public** test subset (500 images). Final ranking uses the **private** subset (1,500 images), revealed after the deadline. Choose which of your submissions is on the leaderboard; that one is ranked.
4. Limits: 3 submissions/day; the whole test set (10,405 questions) must finish within the time limit shown on Codabench (the baseline takes ~6 min on 2 threads); zip ≤ 500 MB.
5. Codabench shows only a **status code**, not logs: 0 ok · 1 files missing · 2 import/`build_model()` failed · 3 `load_state_dict` failed · 4 `predict()` failed on the dummy input or did not return `list[str]` · 5 `predict()` crashed on test data · 6 wrong number/type of answers. Timeouts fail with no status. Debug with `check_submission.py`.

## Rules

1. Inputs are the image tensor and question text only, via the interface above.
2. Train and tune only on the provided **train** and **val** sets. Do not use any other image question–answer annotations.
3. Generic pretrained backbones (ImageNet classifiers, CLIP-style encoders, BERT-family language models, …) are allowed and count toward params and FLOPs. Models or checkpoints trained or fine-tuned for visual question answering, or on any image question–answer data, are **not** allowed.
4. No network calls, external APIs or downloads at inference.
5. Do not try to read, store or transmit test data, and do not hard-code or look up answers. The model must answer from the image and the question.
6. The zip on the leaderboard (with its SHA256) is your frozen entry. On **test day** it is run **offline**, from that same zip (SHA256 checked), on **new images and questions**. Mismatches or rule violations are penalized or disqualified.

## Storage (Google Drive, 15 GB free tier)

The package is **384 MB** (zip). Train images 264 MB, val 36 MB, baseline checkpoint 67 MB.
- Keep only the **zip** on Drive (set `PKG_ZIP` in the notebook). The notebook unpacks it to the Colab disk, which is fast and wiped at the end of the session.
- Feature caches (~0.8 GB) and downloaded pretrained weights stay on the Colab disk, not Drive.
- Keep your **best** checkpoint, not one per epoch, especially for large backbones (a CLIP-based checkpoint is ~350 MB).
- A typical team needs 1–3 GB of Drive.

## Setup outside Colab

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python baseline/train_simple.py
python eval/vqa_eval.py --pred data/preds/simple_val.json --gt data/processed/val/annotations.json
```

## Attribution and licences

- **VQAv2** questions and annotations: © the VQA Consortium, **CC BY 4.0**. Goyal et al., *Making the V in VQA Matter*, CVPR 2017; https://visualqa.org
- **COCO** images: © their Flickr owners, under the licences listed in the COCO annotations; see https://cocodataset.org/#termsofuse. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
- Images here are resized copies (224×224) for course use only. Do not redistribute.
- `eval/normalize.py` ports the answer normalization of the official VQA evaluation code (GT-Vision-Lab/VQA).
