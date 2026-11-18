class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-6):
        self.mean = x.mean()
        self.std = x.std()
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

