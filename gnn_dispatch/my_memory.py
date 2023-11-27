import torch

from gnn_dispatch.models.graph_util import aggr_obs
from jssp_tool.rl.agent.ppo.memory import Memory


class MyMemory(Memory):
    def sample(self, device):
        adjs, features, candidates, masks = self.obs
        adjs = aggr_obs(torch.stack(adjs).to(device), len(adjs[0]))
        features = torch.stack(features).to(device)
        features = features.reshape(-1, features.size(-1))
        candidates = torch.stack(candidates).to(device).squeeze()
        masks = torch.stack(masks).to(device).squeeze()
        obs = (adjs, features, candidates, masks)

        actions = torch.stack(self.actions).to(device).squeeze()
        values = torch.stack(self.values).to(device).squeeze().detach()
        returns = torch.from_numpy(self.returns).to(device).squeeze().detach()
        logprobs = torch.stack(self.logprobs).to(device).squeeze().detach()

        return obs, actions, values, returns, logprobs
