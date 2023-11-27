import os.path
import torch
import numpy as np
import sys

sys.path.append(".")
from gnn_schedule.env.jssp_env import JsspEnv
from gnn_schedule.official.Params import configs
import global_util
from gnn_schedule.models.actor_critic import ActorCritic
from gnn_schedule.runner import Runner
from jssp_tool.env.util import set_random_seed
from jssp_tool.rl.agent.ppo.ppo_discrete import PPODiscrete

configs.device = torch.device(configs.device if torch.cuda.is_available() else "cpu")


def build_ppo(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    return PPODiscrete(
        model,
        configs.eps_clip,
        configs.k_epochs,
        optimizer,
        configs.ploss_coef,
        configs.vloss_coef,
        configs.entloss_coef,
        device=configs.device,
    )


def main():
    set_random_seed(configs.torch_seed)

    env = JsspEnv(configs)
    vali_data = np.load(
        os.path.join(global_util.get_project_root(), "data", "generatedData{}_{}_Seed{}.npy").format(
            configs.n_j, configs.n_m, configs.np_seed_validation
        )
    )

    model = ActorCritic(
        n_j=configs.n_j,
        n_m=configs.n_m,
        num_layers=configs.num_layers,
        learn_eps=False,
        input_dim=configs.input_dim,
        hidden_dim=configs.hidden_dim,
        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic,
        device=configs.device,
    )
    ppo = build_ppo(model)

    # configs.output = os.path.join(
    #     configs.output, "j{}_m{}_l{}_h{}".format(configs.n_j, configs.n_m, configs.low, configs.high)
    # )
    runner = Runner(configs, env, vali_data)
    if configs.test:
        # 加载模型
        ppo.policy.load_state_dict(torch.load(os.path.join(configs.output, "best.pth"), configs.device), False)
        # 测试
        runner.test(vali_data, ppo, float("inf"), 0, phase="test")
    else:
        runner.train(ppo)


if __name__ == "__main__":
    main()
