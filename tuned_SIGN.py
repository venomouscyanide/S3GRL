import torch
from torch_geometric.transforms import SIGN
from torch_sparse import SparseTensor
import torch.nn.functional as F


class TunedSIGN(SIGN):
    """
    Custom SIGN class for SuP and PoS
    """

    def __call__(self, data, sign_k):
        data = super().__call__(data)
        if sign_k == -1:
            for idx in range(1, self.K):
                data.pop(f'x{idx}')
        return data

    def PoS_data_creation(self, pos_data_list):
        original_data = pos_data_list[0]

        for index, data in enumerate(pos_data_list, start=1):
            assert data.edge_index is not None
            row, col = data.edge_index
            adj_t = SparseTensor(row=col, col=row, value=torch.tensor(data.edge_weight),
                                 sparse_sizes=(data.num_nodes, data.num_nodes))

            assert data.x is not None

            original_data[f'x{index}'] = (adj_t @ data.x)
        max_dim = (original_data[f'x'].size()[0])

        for created_op in range(1, self.K + 1):
            max_dim = max(max_dim, original_data[f'x{created_op}'].size()[0])

        for operator in range(1, self.K + 1):
            if original_data[f'x{operator}'].size()[0] == max_dim:
                continue
            original_data[f'x{operator}'] = F.pad(original_data[f'x{operator}'],
                                                  (0, 0, 0, max_dim - original_data[f'x{operator}'].size()[0]))

        if original_data['x'].size()[0] < max_dim:
            original_data['x'] = F.pad(original_data[f'x'],
                                       (0, 0, 0, max_dim - original_data[f'x'].size()[0]))

        # the following keys are useless in SIGN-esque training
        del original_data['node_id']
        del original_data['num_nodes']
        del original_data['edge_index']
        del original_data['edge_weight']

        return original_data
