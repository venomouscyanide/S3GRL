from torch_geometric.transforms import SIGN


class TunedSIGN(SIGN):
    """
    Helps pop the keys that are not consumed during training
    """
    def __call__(self, data, sign_k):
        data = super().__call__(data)
        if sign_k == -1:
            for idx in range(1, self.K):
                data.pop(f'x{idx}')
        return data
