import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from constant import LETTER_LIST


def plot_attention(attention):
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()


# Index conversion
letter2index = { l : i for i, l in enumerate(LETTER_LIST)}
index2letter = { i : l for i, l in enumerate(LETTER_LIST)}


# Decode Utilities
def transcript_to_index(line):
    idxs = [letter2index[l] for l in line]
    idxs.append(letter2index['<eos>'])
    return idxs


def index_to_transcript(indices):
    letters = [index2letter[idx] for idx in indices]
    try:
        ed = letters.index('<eos>')
    except ValueError:
        ed = len(letters)
    letters = letters[:ed]
    return ''.join(letters)


def index_to_transcripts(transcripts):
    processed = []
    for transcript in transcripts:
        transcript = [t.item() for t in transcript]
        processed.append(index_to_transcript(transcript))
    return np.array(processed)


def greedy(probs):
    indices = probs.argmax(dim=1).cpu()
    indices = [t.item() for t in indices]
    return index_to_transcript(indices)


def batch_greedy(probs):
    indices = probs.argmax(dim=2).cpu()
    return index_to_transcripts(indices)


decode = greedy
batch_decode = batch_greedy


# Train Test Split
def random_split(data, proportion=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    nrow = len(data)
    indices = np.random.permutation(nrow)
    test_size = int(nrow * proportion)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    test_data = data.iloc[test_indices]
    train_data = data.iloc[train_indices]
    return train_data, test_data

def K_fold_split(data, k=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    nrow = len(data)
    indices = np.random.permutation(nrow)
    fold_size = nrow // k
    fold_splits = []
    for i in range(k):
        test_mask = np.in1d(range(nrow), indices[i*fold_size:(i+1)*fold_size])
        test = data[test_mask]
        train = data[~test_mask]
        fold_splits.append((train, test))
    return fold_splits

# kf_splits = K_fold_split(processed_df, 5)
# for split in kf_splits:
#     train, val = split
#     print(train.shape, val.shape)
