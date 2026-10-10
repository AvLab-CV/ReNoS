# ReNoS

Code for **Reliable Normality Selection and Scoring for Few-Shot Anomaly Detection**.

## Setup

Create the Conda environment (Python 3.10, PyTorch 2.2, CUDA 11.8):

```bash
conda env create -f environment.yml
conda activate renos
python -m pip install -r requirements.txt
python -m pip install -e .
```

The environment installs the PyTorch CUDA 11.8 runtime. Use a compatible NVIDIA driver.

Download the required datasets separately and set the dataset paths in `foundad/configs/config.yaml` or pass them as Hydra overrides.

## Select support images (CSS)

Run from `foundad/`:

```bash
cd foundad
python final_css.py \
  source=/path/to/dataset \
  target=/path/to/selected_support_images \
  output_csv=/path/to/selection.csv \
  num_samples=1
```

For 2-shot, set:

```bash
num_samples=2 \
selection.stage1_topk=15 \
selection.stage2_topk=10 \
selection.stage3_topk=5
```

For 4-shot, use:

```bash
num_samples=4 \
selection.stage1_topk=20 \
selection.stage2_topk=15 \
selection.stage3_topk=10
```

The selector's `clear_target` option defaults to `true` and clears the target folder before copying selected images.

Return to the repository root before training or evaluation:

```bash
cd ..
```

## Train and evaluate FoundAD

The following example uses Real-IAD 1-shot. FoundAD loads the DINOv3 encoder through `torch.hub`.

```bash
python foundad/main.py mode=train \
  data.dataset=realiad data.data_name=realiad_1shot \
  data.data_path=/path/to/few_shot_data \
  optimization.epochs=8000 app=train_dinov3

python foundad/main.py mode=AD \
  data.dataset=realiad data.data_name=realiad_1shot \
  data.test_root=/path/to/realiad app=test \
  app.ckpt_step=8000 testing.K_top_realiad=5
```

The scripts write checkpoints and evaluation outputs under `logs/`. Check `foundad/configs/config.yaml` for other settings and dataset names.

## License

Upstream license texts for the included FoundAD, DINOv2, and Prismatic code are in `THIRD_PARTY_LICENSES/`. A project license for ReNoS code will be added when selected by the authors.
