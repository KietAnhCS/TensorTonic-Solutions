import numpy as np 

def pad_sequences(seqs, pad_value=0, max_len=None):

    if not seqs: 
        return np.array([]).reshape(0,0).astype(int)

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    if max_len == 0:
        return np.zeros((len(seqs),0), dtype=int)

    num_seqs = len(seqs) 
    padded_array = np.full((num_seqs, max_len), pad_value, dtype=int)

    for i,seq in enumerate(seqs):
        trunc_seq = seq[:max_len]
        padded_array[i, :len(trunc_seq)] = trunc_seq

    return padded_array