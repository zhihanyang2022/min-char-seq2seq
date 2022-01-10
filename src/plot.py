import matplotlib.pyplot as plt
import pickle
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--expdir", type=str, required=True)
args = parser.parse_args()

with open(os.path.join(args.expdir, "train_accs.ob"), 'rb') as fp:
    train_accs = pickle.load(fp)

with open(os.path.join(args.expdir, "valid_accs.ob"), 'rb') as fp:
    valid_accs = pickle.load(fp)

plt.figure(figsize=(5, 5))
plt.plot(train_accs, label="Train")
plt.plot(valid_accs, label="Valid")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Accuracy (Char-Level)")

plt.savefig("training_curve.png", dpi=300)

