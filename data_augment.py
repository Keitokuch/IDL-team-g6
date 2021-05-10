import torchaudio

def specaugment(X, x_len, F=15, mF=2, T=70, p=0.2, mT=2):
        """
        SpecAugment (https://arxiv.org/abs/1904.08779)
        Args:
            x (torch.FloatTensor, [seq_length, dim_features]): The FBANK features.
            F, mF, T, p, mT: The parameters referred in SpecAugment paper.
        """
        x = X[0:x_len, :]
        x = x.T   # [time, freq] --> [freq, time]

        # Freq. masking
        for _ in range(mF):
            x = torchaudio.transforms.FrequencyMasking(F)(x)

        # Time masking
        Tclamp = min(T, int(p * x.shape[1]))
        for _ in range(mT):
            x = torchaudio.transforms.TimeMasking(Tclamp)(x)
        return X