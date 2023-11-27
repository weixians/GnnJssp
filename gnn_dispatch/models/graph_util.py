import torch


def aggr_obs(adj, batch_size):
    """
    聚合多个邻接矩阵为一个大的邻接矩阵，方便整个batch进行矩阵运算
    :param adj:
    :param batch_size:
    :return:
    """
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    idxs = adj.coalesce().indices()
    vals = adj.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * batch_size
    new_idx_col = idxs[2] + idxs[0] * batch_size
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    adj_batch = torch.sparse_coo_tensor(
        indices=idx_mb,
        values=vals,
        size=torch.Size([adj.shape[0] * batch_size, adj.shape[0] * batch_size]),
        dtype=torch.float32,
    ).to(adj.device)
    return adj_batch


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    # batch_size is the shape of batch for graph pool sparse matrix
    if graph_pool_type == "average":
        elem = torch.full(
            size=(batch_size[0] * n_nodes, 1), fill_value=1 / n_nodes, dtype=torch.float32, device=device
        ).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1), fill_value=1, dtype=torch.float32, device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0], device=device, dtype=torch.long)
    # print(idx_0)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0] * n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes * batch_size[0], device=device, dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse_coo_tensor(
        idx, elem, torch.Size([batch_size[0], n_nodes * batch_size[0]]), dtype=torch.float32
    ).to(device)
    return graph_pool
