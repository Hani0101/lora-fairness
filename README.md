# LoRA reproduction + fairness comparison

From-scratch PyTorch implementation of LoRA (Hu et al., 2021, ICLR 2022),
plus a full fine-tuning baseline, reproducing the paper's accuracy-efficiency
tradeoff on RoBERTa-base and extending it with a fairness comparison using
counterfactual data augmentation and CrowS-Pairs.

## Layout

```
src/lora/layers.py   LoRALinear: h = W0 x + (alpha/r) B A x, merge/unmerge
src/lora/inject.py   swap targeted nn.Linear -> LoRALinear, freeze, count params
src/data.py          GLUE task registry + dataloaders
src/metrics.py       accuracy / F1 / MCC / correlation, selected per task
src/train.py         one training loop, three methods (lora | full | head_only)
tests/test_lora.py   invariants that catch silent LoRA bugs

src/fairness/cda.py            counterfactual data augmentation (gender word-swap)
src/fairness/crows_pairs.py    CrowS-Pairs loading + pseudo-log-likelihood bias scoring
src/fairness/train_mlm.py      continued MLM pretraining on CDA-augmented WikiText-2
src/fairness/evaluate_bias.py  scores base / LoRA / full checkpoints on CrowS-Pairs
tests/test_fairness.py         augmentation + scoring-mechanics tests

notebooks/colab_lora_reproduction.ipynb   run the accuracy sweep on a Colab T4
notebooks/colab_fairness_benchmark.ipynb  run the fairness comparison on a Colab T4

report_polished.md, report_compressed.md   written report drafts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

## Running

```bash
# Smoke test on CPU, 200 training examples, 1 epoch
python -m src.train --method lora --task mrpc --model distilbert-base-uncased \
    --train-subset 200 --epochs 1

# Paper-style MRPC comparison, repeated over seeds
for seed in 0 1 2; do
  python -m src.train --method lora --task mrpc --model roberta-base --seed $seed
  python -m src.train --method full --task mrpc --model roberta-base --seed $seed
done
```

Each run writes `runs/<task>_<model>_<method>_seed<n>/results.json` with the
parameter counts, per-epoch metrics, best epoch, and wall clock.

## Defaults and where they come from

| Setting | Value | Source |
|---|---|---|
| Adapted weights | `Wq`, `Wv` only | Section 4.2, Table 5 |
| Rank `r` | 8 | Table 9 (RoBERTa base) |
| `alpha` | 8 | Table 9 (RoBERTa base) |
| LR (LoRA) | 4e-4 | Table 9, MRPC column |
| LR (full FT) | 2e-5 | standard RoBERTa fine-tuning |
| Warmup ratio | 0.06 | Table 9 |
| Max seq length | 128 | Section 5.2, adapter-comparable setup |

Two deliberate deviations from the paper, both worth stating in the report:

1. The paper initializes MRPC runs from an MNLI checkpoint. Here both methods
   start from the pretrained base, which lowers absolute accuracy but keeps
   the LoRA vs full FT comparison clean.
2. `A` is initialized `N(0, 0.02)` per the paper text. The reference repo uses
   Kaiming uniform. Pass `init_std=None` in `LoRAConfig` to switch.

The classification head is randomly initialized and is therefore trained under
every method, including LoRA. It is counted in `trainable` but reported
separately as `lora_only` so the parameter-efficiency claim stays honest.

## Baselines

`--method head_only` freezes the entire encoder and trains only the classifier.
Not in the paper, but it is the floor the LoRA number has to clear to mean
anything. Worth one run.

## GLUE reproduction results

RoBERTa-base on MRPC, mean over 3 seeds (0, 1, 2), Google Colab NVIDIA T4:

| Source | Method | Trainable params | % of total | Accuracy | F1 | Time/run |
|---|---|---|---|---|---|---|
| Hu et al. (2021), Table 2 | Full fine-tuning | 125.0M | 100% | 90.2% | -- | -- |
| Hu et al. (2021), Table 2 | LoRA (r=8) | 0.3M | ~0.24% | 89.7% | -- | -- |
| This project | Full fine-tuning | 124,647,170 | 100% | 88.9% | 92.1% | ~595s |
| This project | LoRA (r=8) | 887,042 | 0.71% | 88.2% | 91.5% | ~360s |

Absolute accuracy sits 1.5 to 2 points below the paper's, most likely because
the paper initializes MRPC from an MNLI-adapted checkpoint while this project
starts from the base pretrained model. The relative gap between methods
reproduces closely: under one point separates LoRA from full fine-tuning in
both cases, with LoRA using orders of magnitude fewer parameters. LoRA showed
wider seed variance (87.0 to 89.7% accuracy) than full fine-tuning (88.7 to
89.0%).

## Fairness benchmark

Separate from the GLUE reproduction above: this compares whether a base
pretrained model, a LoRA-adapted model, and a fully fine-tuned model differ
in gender-stereotype bias, as measured by CrowS-Pairs.

**Why this is a separate pipeline, not built on the MRPC checkpoints.**
CrowS-Pairs scores sentences via a masked-language-model head. The MRPC
checkpoints are `RobertaForSequenceClassification`, and the MLM head was
discarded and replaced with a randomly-initialized classifier during that
training. So this pipeline runs its own training step, on `RobertaForMaskedLM`
throughout (training and scoring), so there is no head mismatch.

**Pipeline:**
1. `src/fairness/cda.py`: counterfactual data augmentation. Swaps gendered
   terms (he/she, actor/actress, and around 40 word pairs total) in a
   sentence to produce a balanced duplicate. Gender only, see limitations
   below.
2. `src/fairness/train_mlm.py`: continues MLM pretraining on a CDA-augmented
   slice of WikiText-2, via `--method lora` or `--method full`.
3. `src/fairness/crows_pairs.py`: loads CrowS-Pairs (1,508 held-out sentence
   pairs, never used in training) and scores a model via pseudo-log-likelihood
   (Nangia et al., 2020): for each pair, mask each token shared between the
   two sentences one at a time, sum the model's log-probability of the true
   token, and see which sentence scores higher.
4. `src/fairness/evaluate_bias.py`: runs that scoring across a fresh base
   model, the LoRA checkpoint, and the full-FT checkpoint, and writes a
   comparison table.

**Running it (Colab):**
```bash
python -m src.fairness.train_mlm --method lora --model roberta-base --seed 0
python -m src.fairness.train_mlm --method full --model roberta-base --seed 0
python -m src.fairness.evaluate_bias --model roberta-base \
    --lora-checkpoint runs_fairness/cda_roberta-base_lora_seed0/lora_weights.pt \
    --full-checkpoint runs_fairness/cda_roberta-base_full_seed0/model.pt
```
Or use `notebooks/colab_fairness_benchmark.ipynb` directly.

**Corpus size.** `--n-lines` subsamples WikiText-2 before augmentation.
Pass `--n-lines 0` to use the full filtered corpus with no subsampling
(the flag translates 0, or any value at or below 0, to "no subsampling"
internally). Two corpus sizes were tested for this project: a 5,000-line
subsample and the full corpus.

**Metric:** `pct_stereotype`, the percentage of CrowS-Pairs pairs where the
model preferred the more-stereotypical sentence. 50% is the unbiased
baseline; higher means more stereotype-favoring.

**Results.**

Training cost, full corpus:

| Method | Trainable params | % of total | Wall-clock |
|---|---|---|---|
| LoRA (r=8) | 294,912 | 0.24% | 1,940.9s (~32 min) |
| Full fine-tuning | 124,697,433 | 100% | 2,983.8s (~50 min) |

CrowS-Pairs bias comparison:

| Corpus | Model | Overall pct_stereotype | Gender-only pct_stereotype |
|---|---|---|---|
| -- | Base (no fine-tuning) | 59.35% | 54.96% |
| 5,000 lines | LoRA | 60.61% | 56.11% |
| 5,000 lines | Full fine-tuning | 58.95% | 55.73% |
| Full corpus | LoRA | 60.68% | 56.87% |
| Full corpus | Full fine-tuning | 59.55% | 53.82% |

CDA did not reduce gender-stereotype preference under LoRA at either corpus
size, it increased in both cases. Full fine-tuning's effect depended on
corpus size: a small increase at 5,000 lines, but a decrease below the base
model on the full corpus, the only configuration across every run where
fine-tuning moved the model closer to unbiased than it started.

For more information about the results and the experiment please refer to lora-fariness.docx
