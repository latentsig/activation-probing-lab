# Activation Probing Lab

A small, reproducible companion project for monitoring what a 4B language model learns during QLoRA fine-tuning.

The lab tracks three views of the same run:

1. the training objective;
2. target information that can be decoded from checkpoint activations;
3. a deliberately planted shortcut that the model may learn instead.

It is designed to make the workflow in [Latent Signals: what model internals tell us before the loss curve does](https://latentsig.com/insights/latent-signals-activation-probing/) concrete enough to run on one GPU.

## What this project does

The included toy task teaches `Qwen/Qwen3-4B` an arbitrary routing code. The correct answer depends on a combination of two fields. A source tag agrees with the answer 95 percent of the time in the fine-tuning set, which gives the model an easier but brittle shortcut.

The probe set breaks that correlation and adds a new surface template. At every saved checkpoint, the lab captures the final prompt-token residual stream at 25, 50, 75, and 100 percent of model depth. It then measures:

- target AUROC on held-out and transfer examples;
- shortcut AUROC on the same examples;
- repeated label-permutation controls;
- grouped cross-validation with an L2-regularized linear probe;
- bootstrap confidence intervals;
- a step-0 probe transferred forward, which exposes direction stability or rotation.

This is a diagnostic, not a claim that a decoded feature is causally used by the model.

## Model and hardware

The default model is [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), an Apache-2.0 causal language model with 4.0B parameters and 36 transformer layers. Qwen3 requires Transformers 4.51 or newer.

The training path uses NF4 with double quantization and LoRA adapters. A CUDA GPU with at least 16 GB of memory is a sensible starting point for the default sequence length and batch settings. Exact memory use depends on the CUDA, PyTorch, Transformers, PEFT, and bitsandbytes versions on the machine.

No GPU is required for the smoke demo or unit tests.

## Quick start: CPU smoke demo

The smoke demo creates synthetic checkpoint activations with a known target and shortcut direction. It runs the exact same probe and plotting code used by the 4B experiment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

apl --config configs/qwen3-4b-toy.yaml smoke-demo
```

Open:

```text
runs/smoke-demo/report/probe_trajectories.png
```

![Synthetic example showing loss, target decodability, and shortcut decodability](docs/example-probe-trajectories.png)

This checked-in image comes from synthetic activations, not a Qwen3-4B result. Its purpose is to verify the analysis pipeline and show how the final report is structured.

## Run the 4B experiment

Install the GPU dependencies:

```bash
python -m pip install -e '.[train]'
```

Generate the task:

```bash
apl --config configs/qwen3-4b-toy.yaml generate-data
```

Fine-tune the adapter and save checkpoints at steps 20, 40, 60, 80, and 100:

```bash
apl --config configs/qwen3-4b-toy.yaml train
```

Capture activations for the base model and every saved adapter checkpoint:

```bash
apl --config configs/qwen3-4b-toy.yaml capture
```

Fit the probe panel and render the report:

```bash
apl --config configs/qwen3-4b-toy.yaml probe
```

The main outputs are:

```text
runs/qwen3-4b-toy/report/probe_results.csv
runs/qwen3-4b-toy/report/probe_trajectories.png
runs/qwen3-4b-toy/report/probe_trajectories.svg
```

## Reading the plot

Each color corresponds to a normalized model depth. Solid lines refit a regularized probe at every checkpoint. Dotted lines keep the step-0 probe fixed and apply it to later checkpoints.

A useful learning story looks like this:

- target transfer AUROC rises across checkpoints;
- shortcut AUROC does not dominate the target signal;
- the mean permutation control remains close to chance;
- the fixed step-0 direction and refit probe tell a consistent story.

A warning pattern is target performance rising in-domain while remaining flat on the transfer split, especially when shortcut AUROC rises quickly. That is evidence to inspect the data or training setup, not a reason to declare the run successful because loss fell.

## Experiment layout

```text
generated task
    |
    v
Qwen3-4B + QLoRA checkpoints
    |
    v
fixed prompts + fixed token site + fixed layer fractions
    |
    v
target probe + shortcut probe + controls
    |
    v
checkpoint trajectory and decision evidence
```

## Why the controls matter

The hidden size of Qwen3-4B is 2,560. A small dataset can produce an attractive separating direction through noise alone. This project tunes the ridge strength inside grouped cross-validation and reports a label-permutation baseline.

The target and shortcut labels are balanced independently in every probe split. The transfer split also uses unseen prose templates. These choices make it harder for the probe itself to win by exploiting the same surface correlation planted in the training set.

## Adapt it to your data

The training JSONL schema is:

```json
{"prompt":"...","response":"KITE"}
```

The probe JSONL adds measurement fields:

```json
{
  "id": "probe-00001",
  "prompt": "...",
  "target": 1,
  "shortcut": 0,
  "split": "probe_transfer",
  "group": "source-or-template-family"
}
```

Keep the probe prompts, splits, token site, layer sites, and regularization grid frozen before comparing checkpoints. Replace `target` and `shortcut` with concepts that have operational meaning for your run.

## Tests

```bash
make test
```

The tests do not download a language model. They validate the toy-data balance, layer mapping, checkpoint ordering, regularized probe recovery, and bootstrap interval code.

## References

- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [Transformers bitsandbytes and QLoRA documentation](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/package_reference/lora)
- [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
- [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## License

MIT. See [LICENSE](LICENSE).
