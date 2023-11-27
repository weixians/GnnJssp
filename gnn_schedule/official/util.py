import numpy as np
import torch


def to_tensor(adj, fea, candidate, mask, device):
    precedent_matrix, succedent_matrix, disjunctive_matrix, all_item_matrix = adj
    precedent_matrix_tensor = torch.from_numpy(np.copy(precedent_matrix)).to(device).to_sparse()
    succedent_matrix_tensor = torch.from_numpy(np.copy(succedent_matrix)).to(device).to_sparse()
    disjunctive_matrix_tensor = torch.from_numpy(np.copy(disjunctive_matrix)).to(device).to_sparse()
    all_item_matrix_tensor = torch.from_numpy(np.copy(all_item_matrix)).to(device).to_sparse()
    adj_tensor_tuple = (
        precedent_matrix_tensor,
        succedent_matrix_tensor,
        disjunctive_matrix_tensor,
        all_item_matrix_tensor,
    )
    fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
    candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device).unsqueeze(0)
    mask_tensor = torch.from_numpy(np.copy(mask)).to(device).unsqueeze(0)
    return adj_tensor_tuple, fea_tensor, candidate_tensor, mask_tensor
