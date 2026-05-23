"""RELIEF PPO ensemble agent (ActorDiscrete + ActorContinuous + Critic + H_PPO).

Copied verbatim from the original RELIEF repository, adapted for PyG data objects.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def orthogonal_init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


def xavier_init(layer):
    nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("relu"))
    nn.init.constant_(layer.bias, 0)


class ActorDiscrete(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim * 0.5)),
            nn.Tanh(),
            nn.Linear(int(hid_dim * 0.5), int(hid_dim * 0.25)),
            nn.Tanh(),
            nn.Linear(int(hid_dim * 0.25), 1),
        )
        orthogonal_init(self.linears[0])
        orthogonal_init(self.linears[2])
        orthogonal_init(self.linears[4], gain=0.01)

    def forward(self, state, mask):
        output = self.linears(state).squeeze(-1)
        output = torch.where(mask == 0, output, torch.full_like(output, float("-inf")))
        action_prob = F.softmax(output, dim=-1)
        return action_prob

    def get_action_logprob(self, state, mask):
        action_prob = self.forward(state, mask)
        dist = Categorical(action_prob)
        action_d = dist.sample()
        logprob = dist.log_prob(action_d)
        return action_d, logprob

    def pi(self, state, mask, require_grad=False):
        if require_grad:
            action_prob = self.forward(state, mask)
        else:
            with torch.no_grad():
                action_prob = self.forward(state, mask)
        return action_prob

    def logprob_entropy(self, state, action_d, mask):
        action_prob = self.forward(state, mask)
        dist = Categorical(action_prob)
        return dist.log_prob(action_d), dist.entropy()


class ActorContinuous(nn.Module):
    def __init__(self, hid_dim, x_dim, max_action_con, init_log_std):
        super().__init__()
        self.max_action = max_action_con
        self.log_std = torch.ones(x_dim) * init_log_std
        self.linears = nn.Sequential(
            nn.Linear(hid_dim, int(hid_dim * 0.5)),
            nn.Tanh(),
            nn.Linear(int(hid_dim * 0.5), int(hid_dim * 0.5)),
            nn.Tanh(),
            nn.Linear(int(hid_dim * 0.5), x_dim),
        )
        orthogonal_init(self.linears[0])
        orthogonal_init(self.linears[2])
        orthogonal_init(self.linears[4], gain=0.01)

    def forward(self, state, action_d):
        output = self.linears(state)
        index = action_d.view(action_d.size(0), 1, 1).expand(action_d.size(0), 1, output.size(-1))
        output = torch.gather(output, dim=1, index=index).squeeze(1)
        mean = torch.tanh(output) * self.max_action
        return mean

    def get_action_logprob(self, state, action_d, deterministic=False):
        mean = self.forward(state, action_d)
        std = torch.exp(self.log_std.expand_as(mean).to(mean.device))
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        action = torch.clamp(action, -self.max_action, self.max_action)
        logprob = torch.sum(dist.log_prob(action), dim=-1)
        return action, logprob

    def logprob_entropy(self, state, action_d, action_c):
        mean = self.forward(state, action_d)
        std = torch.exp(self.log_std.expand_as(mean).to(mean.device))
        dist = Normal(mean, std)
        return torch.sum(dist.log_prob(action_c), dim=-1), torch.sum(dist.entropy(), dim=-1)


class Critic(nn.Module):
    def __init__(self, hid_dim, max_num_nodes):
        super().__init__()
        input_dim = int(hid_dim * max_num_nodes)
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * 0.05)),
            nn.ReLU(),
            nn.Linear(int(input_dim * 0.05), int(input_dim * 0.025)),
            nn.ReLU(),
            nn.Linear(int(input_dim * 0.025), 1),
        )
        xavier_init(self.linears[0])
        xavier_init(self.linears[2])
        xavier_init(self.linears[4])

    def forward(self, state):
        return self.linears(self.flatten(state)).squeeze(-1)


class H_PPO:
    """RELIEF hierarchical PPO agent with ensemble actors."""

    def __init__(
        self,
        gnn,
        ensemble_num,
        penalty_alpha_d,
        penalty_alpha_c,
        hid_dim,
        x_dim,
        max_num_nodes,
        max_action,
        init_log_std,
        eps_clip_d,
        eps_clip_c,
        coeff_critic,
        coeff_entropy_d,
        max_norm_grad,
        device,
    ):
        self.gnn = gnn
        self.ensemble_num = ensemble_num
        self.penalty_alpha_d = penalty_alpha_d
        self.penalty_alpha_c = penalty_alpha_c
        self.actor_ds = [
            copy.deepcopy(ActorDiscrete(hid_dim).to(device)) for _ in range(ensemble_num)
        ]
        self.actor_cs = [
            copy.deepcopy(ActorContinuous(hid_dim, x_dim, max_action, init_log_std).to(device))
            for _ in range(ensemble_num)
        ]
        self.critic = Critic(hid_dim, max_num_nodes).to(device)

        self.eps_clip_d = eps_clip_d
        self.eps_clip_c = eps_clip_c
        self.coeff_critic = coeff_critic
        self.coeff_ent_d = coeff_entropy_d
        self.max_norm_grad = max_norm_grad
        self.max_num_nodes = max_num_nodes
        self.device = device

        self.prompt_slices = []
        self.step_state = None
        self.batch_mask = None

    def train_or_eval(self, mode):
        if mode == "train":
            for i in range(self.ensemble_num):
                self.actor_ds[i].train()
                self.actor_cs[i].train()
            self.critic.train()
        elif mode == "eval":
            for i in range(self.ensemble_num):
                self.actor_ds[i].eval()
                self.actor_cs[i].eval()
            self.critic.eval()
        else:
            raise ValueError("Invalid model mode! Choose between 'train' and 'eval'.")

    def _node_to_state(self, batch_data, prompt, nodes_per_graph):
        with torch.no_grad():
            # NOTE: pass prompt/ batch as kwargs because GNN.forward signature
            # is (x, edge_index, batch=None, prompt=None, prompt_type=None, ...);
            # positional ``prompt`` here would land in ``batch`` and trigger
            # ``self.pool(node_emb, batch.long())`` with a 2-D tensor.
            node_emb = self.gnn(batch_data.x, batch_data.edge_index, prompt=prompt)
        state_slices = torch.split(node_emb, nodes_per_graph.tolist(), dim=0)
        for i, ss in enumerate(state_slices):
            self.step_state[i, : ss.size(0), :] = ss
        return self.step_state

    def train_prompt(self, a_idx, batch_data, nodes_per_graph, step):
        with torch.no_grad():
            self.gnn.eval()
            batch_state = self._node_to_state(
                batch_data, torch.cat(self.prompt_slices), nodes_per_graph
            )
            batch_action_d, batch_logp_d = self.actor_ds[a_idx].get_action_logprob(
                batch_state, self.batch_mask
            )
            batch_action_c, batch_logp_c = self.actor_cs[a_idx].get_action_logprob(
                batch_state, batch_action_d
            )
            for i in range(len(self.prompt_slices)):
                if step < nodes_per_graph[i]:
                    self.prompt_slices[i][batch_action_d[i]] += batch_action_c[i]
        return batch_state, batch_action_d, batch_action_c, batch_logp_d, batch_logp_c

    def _get_eval_action_d(self, batch_state, batch_mask):
        pis = []
        for i in range(self.ensemble_num):
            pis.append(self.actor_ds[i].pi(batch_state, batch_mask, require_grad=False))
        ens_max_pi = torch.stack(pis, dim=1).max(dim=1)[0]
        ens_max_prob = F.normalize(ens_max_pi + 1e-10, p=1, dim=-1)
        dist = Categorical(ens_max_prob)
        return dist.sample()

    def _get_eval_action_c(self, batch_state, batch_action_d):
        means = []
        for i in range(self.ensemble_num):
            with torch.no_grad():
                mean = self.actor_cs[i].forward(batch_state, batch_action_d)
            means.append(mean)
        ens_mean = torch.stack(means, dim=1).mean(dim=1)
        return ens_mean

    def eval_prompt(self, batch_data, nodes_per_graph, step, truncate_flag):
        with torch.no_grad():
            self.gnn.eval()
            batch_state = self._node_to_state(
                batch_data, torch.cat(self.prompt_slices), nodes_per_graph
            )
            batch_action_d = self._get_eval_action_d(batch_state, self.batch_mask)
            batch_action_c = self._get_eval_action_c(batch_state, batch_action_d)
        for i in range(len(self.prompt_slices)):
            if step < nodes_per_graph[i] and not truncate_flag[i]:
                if batch_action_d[i] < nodes_per_graph[i]:
                    self.prompt_slices[i][batch_action_d[i]] += batch_action_c[i]

    def _ensemble_penalty_d(self, a_idx, state, mask):
        pis = []
        for i in range(self.ensemble_num):
            pi = self.actor_ds[i].pi(state, mask, require_grad=True if i == a_idx else False)
            pis.append(pi)
        with torch.no_grad():
            ens_max_pi = torch.stack(pis, dim=1).max(dim=1)[0]
            ens_pi = F.normalize(ens_max_pi, p=1, dim=-1)
        cur_pi = F.normalize(pis[a_idx] + 1e-10, p=1, dim=-1)
        ens_pi = F.normalize(ens_pi + 1e-10, p=1, dim=-1)
        kl_penalty_d = F.kl_div(torch.log(cur_pi), ens_pi, log_target=False, reduction="none").sum(
            dim=-1
        )
        ens_action_d = torch.argmax(ens_max_pi, dim=-1)
        return kl_penalty_d * self.penalty_alpha_d, ens_action_d

    def _ensemble_penalty_c(self, a_idx, state, action_d):
        means, stds = [], []
        for i in range(self.ensemble_num):
            if i == a_idx:
                mean = self.actor_cs[i].forward(state, action_d)
            else:
                with torch.no_grad():
                    mean = self.actor_cs[i].forward(state, action_d)
            std = torch.exp(self.actor_cs[i].log_std.expand_as(mean).to(mean.device))
            means.append(mean)
            stds.append(std)
        with torch.no_grad():
            ens_mean = torch.stack(means, dim=1).mean(dim=1)
            ens_std = torch.stack(stds, dim=1).mean(dim=1)
        cur_dist = Normal(means[a_idx], stds[a_idx])
        ens_dist = Normal(ens_mean, ens_std)
        kl_penalty_c = torch.distributions.kl.kl_divergence(cur_dist, ens_dist).sum(dim=-1)
        return kl_penalty_c * self.penalty_alpha_c

    def _actor_loss(
        self, a_idx, state, action_d, action_c, adv, logprob_old_d, logprob_old_c, node_num
    ):
        base_mask = torch.zeros(state.size(0), self.max_num_nodes).to(self.device)
        base_idx = torch.arange(0, self.max_num_nodes).to(self.device)
        mask = torch.where(base_idx < node_num.unsqueeze(-1), base_mask, torch.ones_like(base_mask))

        logprob_new_d, entropy_d = self.actor_ds[a_idx].logprob_entropy(state, action_d, mask)
        ratio_d = torch.exp(logprob_new_d - logprob_old_d)
        surr1_d = ratio_d * adv
        surr2_d = torch.clamp(ratio_d, 1 - self.eps_clip_d, 1 + self.eps_clip_d) * adv
        if self.ensemble_num > 1 and self.penalty_alpha_d > 0:
            ens_pen_d, ens_action_d = self._ensemble_penalty_d(a_idx, state, mask)
        else:
            ens_pen_d = torch.zeros(1).to(self.device)
            ens_action_d = None
        actor_loss_d = -torch.min(surr1_d, surr2_d) - self.coeff_ent_d * entropy_d + ens_pen_d
        with torch.no_grad():
            approx_kl_d = ((ratio_d - 1) - (logprob_new_d - logprob_old_d)).mean()

        logprob_new_c, entropy_c = self.actor_cs[a_idx].logprob_entropy(state, action_d, action_c)
        ratio_c = torch.exp(logprob_new_c - logprob_old_c)
        surr1_c = ratio_c * adv
        surr2_c = torch.clamp(ratio_c, 1 - self.eps_clip_c, 1 + self.eps_clip_c) * adv
        if self.ensemble_num > 1 and self.penalty_alpha_c > 0:
            ens_pen_c = self._ensemble_penalty_c(a_idx, state, ens_action_d)
        else:
            ens_pen_c = torch.zeros(1).to(self.device)
        actor_loss_c = -torch.min(surr1_c, surr2_c) + ens_pen_c
        with torch.no_grad():
            approx_kl_c = ((ratio_c - 1) - (logprob_new_c - logprob_old_c)).mean()

        return (
            actor_loss_d.mean(),
            actor_loss_c.mean(),
            entropy_d.mean().item(),
            entropy_c.mean().item(),
            ens_pen_d.mean().item(),
            ens_pen_c.mean().item(),
            approx_kl_d.item(),
            approx_kl_c.item(),
        )

    def _critic_loss(self, state, ret):
        state_value = self.critic(state).squeeze(-1)
        return F.mse_loss(ret.float(), state_value.float()) * self.coeff_critic

    def train_policy(
        self, args, a_idx, experience, node_num, optim, ret0, scaled_reward, batch_id, total_batch
    ):
        state, action_d, logprob_old_d, action_c, logprob_old_c, adv, ret = experience
        node_num = torch.stack([x for x in node_num for _ in range(x)])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_loss_d_batch = actor_loss_c_batch = critic_loss_batch = 0.0
        ens_pen_d_batch = ens_pen_c_batch = kl_d_batch = kl_c_batch = 0.0

        minibatch_num = int(len(state) / args.minibatch_size)
        update_num_cnt = 0
        for _ in range(args.policy_update_nums):
            index = torch.randperm(len(state))
            kl_d_minibatch = kl_c_minibatch = 0.0
            for minibatch in range(minibatch_num):
                idx = index[minibatch * args.minibatch_size : (minibatch + 1) * args.minibatch_size]
                actor_loss_d, actor_loss_c, _, _, ens_pen_d, ens_pen_c, kl_d, kl_c = (
                    self._actor_loss(
                        a_idx,
                        state[idx],
                        action_d[idx],
                        action_c[idx],
                        adv[idx],
                        logprob_old_d[idx],
                        logprob_old_c[idx],
                        node_num[idx],
                    )
                )
                kl_d_minibatch += kl_d
                kl_c_minibatch += kl_c
                critic_loss = self._critic_loss(state[idx], ret[idx])

                optim.zero_grad()
                loss = actor_loss_d + actor_loss_c + critic_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_ds[a_idx].parameters())
                    + list(self.actor_cs[a_idx].parameters())
                    + list(self.critic.parameters()),
                    norm_type=2,
                    max_norm=self.max_norm_grad,
                )
                optim.step()

                if not getattr(args, "sh_mode", False):
                    actor_loss_d_batch += actor_loss_d.item()
                    actor_loss_c_batch += actor_loss_c.item()
                    critic_loss_batch += critic_loss.item()
                    ens_pen_d_batch += ens_pen_d
                    ens_pen_c_batch += ens_pen_c
                    kl_d_batch += kl_d
                    kl_c_batch += kl_c

            update_num_cnt += 1
            if getattr(args, "target_kl_d", None) is not None:
                if kl_d_minibatch / minibatch_num > args.target_kl_d:
                    break
            if getattr(args, "target_kl_c", None) is not None:
                if kl_c_minibatch / minibatch_num > args.target_kl_c:
                    break

        if not getattr(args, "sh_mode", False):
            print(
                f"[Actor{a_idx + 1} Batch{batch_id:02d}/{total_batch}] "
                f"Loss {actor_loss_d_batch / minibatch_num / update_num_cnt:.3f} "
                f"{actor_loss_c_batch / minibatch_num / update_num_cnt:.3f} "
                f"{critic_loss_batch / minibatch_num / update_num_cnt:.2f}"
            )
