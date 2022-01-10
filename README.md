<h1 align="center">Minimal character-level seq2seq</h1>
<p align="center">An well-styled implementation in PyTorch for learners. Converge within minutes for toy datasets.</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/43589364/148682298-1b3a8e62-05d3-442d-ba94-d44e5ff09cf8.png" width=600>
</p>

## Requirements

```bash
pip install gin-config torch numpy matplotlib
```

## Datasets

- Reversing 4 digits
- Reversing 20 digits
- Summing 2 integers

## Scripts

Dataset generation example:

```bash
cd src/datasets
mkdir reverse_int_long
python generate_dataset_reverse_int_short.py
```

Model training example:

```bash
cd src
python train.py --expdir=reverse_int_long_with_attention
```

## Visualization

You can generate the attention matrix of a random example in the test set using:

```bash
python train.py --expdir=reverse_int_long_with_attention --infer
```

You can find attention matrix inside the experiment dir afterwards:

<img src="src/experiments/reverse_int_long_with_attention/attention_matrix.png" width=400>
