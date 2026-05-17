"""RELIEF training and evaluation loop.

Encapsulates attach_prompt, train_policy, evaluate_policy and helpers.
"""

from __future__ import annotations

import copy

import torch
from torch_geometric.data import Batch
from torch_scatter import scatter

from .reward import Normalization, compute_adv_ret, reshape_reward, reward_reshape

_criterion = torch.nn.CrossEntropyLoss(reduction="none")


def attach_prompt(args, policy, data, data_type, gnn=None, tasknet=None, compute_reward=0, compute_prr=0):
    device = data.x.device
    graph_num = data.num_graphs if hasattr(data, "num_graphs") else 1
    if graph_num == 1:
        nodes_per_graph = torch.tensor([data.num_nodes], device=device)
    else:
        nodes_per_graph = scatter(torch.ones_like(data.batch), data.batch, reduce="add")

    policy.prompt_slices = [torch.zeros(nodes, args.svd_dim).to(device) for nodes in nodes_per_graph]
    policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.hid_dim).to(device)
    base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
    base_idx = torch.arange(0, policy.max_num_nodes).to(device)
    policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))

    if compute_reward:
        init_r = generate_reward(gnn, tasknet, data)
        r_list = [copy.deepcopy([]) for _ in range(graph_num)]
        for rl, ir in zip(r_list, init_r):
            rl.append(ir)

    truncate_flag = torch.zeros(graph_num, dtype=torch.bool, device=device)
    batch_max_step = nodes_per_graph.max().item()
    for step in range(batch_max_step):
        if truncate_flag.sum() == graph_num:
            break
        policy.eval_prompt(data, nodes_per_graph, step, truncate_flag)
        if compute_reward:
            rs = generate_reward(gnn, tasknet, data, torch.cat(policy.prompt_slices))
            for i in range(graph_num):
                if step < nodes_per_graph[i] and not truncate_flag[i]:
                    r_list[i].append(rs[i])

    reward_list = None
    pr_ratio_list = None
    if compute_reward:
        reward_list = [reward_reshape(torch.stack(r).cpu().numpy(), reward_clip=args.reward_clip) for r in r_list]
    if compute_prr:
        pr_ratio_list = [torch.any(p != 0, dim=-1).sum().item() / p.size(0) for p in policy.prompt_slices]

    return torch.cat(policy.prompt_slices), reward_list, pr_ratio_list


def generate_reward(gnn, tasknet, data, prompt=None):
    with torch.no_grad():
        graph_emb = gnn(data.x, data.edge_index, prompt, getattr(data, "batch", None))
        logit = tasknet(graph_emb)
    return _criterion(logit, data.y)


def tasknet_loss(gnn, tasknet, data, prompt=None, require_grad=True, keep_loss_dim=False, policy_gnn_update=False):
    if policy_gnn_update:
        gnn.train()
    with torch.no_grad():
        graph_emb = gnn(data.x, data.edge_index, prompt, getattr(data, "batch", None))
    if require_grad:
        logit = tasknet(graph_emb)
    else:
        with torch.no_grad():
            logit = tasknet(graph_emb)
    loss = _criterion(logit, data.y)
    if not keep_loss_dim:
        loss = torch.mean(loss)
    if policy_gnn_update:
        gnn.eval()
    return loss, logit.detach().cpu(), data.y.cpu()


def evaluate_tasknet(gnn, tasknet, loader, device):
    gnn.eval()
    tasknet.eval()
    epoch_loss = 0.0
    preds, ys = [], []
    for batch_data in loader:
        batch_data = batch_data.to(device)
        loss, logit, y = tasknet_loss(gnn, tasknet, batch_data, prompt=None, require_grad=False)
        pred = logit.argmax(dim=1)
        preds.append(pred)
        ys.append(y)
        epoch_loss += loss.item()
    epoch_loss /= len(loader)
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    acc = (preds == ys).float().mean().item() * 100
    return epoch_loss, acc


def evaluate_policy(args, gnn, tasknet, policy, loader, data_type, device):
    gnn.eval()
    tasknet.eval()
    policy.train_or_eval(mode="eval")
    pr_ys, pr_preds, pr_losses = [], [], []
    for batch_data in loader:
        batch_data = batch_data.to(device)
        prompt, _, _ = attach_prompt(args, policy, batch_data, data_type, gnn, tasknet, compute_reward=0, compute_prr=0)
        pr_loss, pr_logit, pr_y = tasknet_loss(gnn, tasknet, batch_data, prompt, require_grad=False, keep_loss_dim=True)
        pr_ys.append(pr_y)
        pr_preds.append(pr_logit.argmax(dim=1))
        pr_losses.append(pr_loss.cpu())
    ys = torch.cat(pr_ys)
    preds = torch.cat(pr_preds)
    losses = torch.cat(pr_losses).mean().item()
    acc = (preds == ys).float().mean().item() * 100
    return acc, losses


def train_policy_epoch(args, epoch, gnn, tasknet, tasknet_optim, policy, policy_optims, ens_loaders, train_loader, val_loader, test_loader, device):
    descend_decay_frac = 1.0 - (epoch - 1) / args.total_epochs
    policy_decay_frac = 1.0
    if getattr(args, "policy_decay", "") == "down":
        policy_decay_frac = descend_decay_frac
    for i in range(args.ensemble_num):
        for param_group in policy_optims[i].param_groups:
            if param_group["name"] == "actor_d":
                param_group["lr"] = policy_decay_frac * args.actor_d_lr
            elif param_group["name"] == "actor_c":
                param_group["lr"] = policy_decay_frac * args.actor_c_lr
            elif param_group["name"] == "critic":
                param_group["lr"] = policy_decay_frac * args.critic_lr

    tasknet_decay_frac = 1.0
    if getattr(args, "tasknet_decay", "") == "down":
        tasknet_decay_frac = descend_decay_frac
    for param_group in tasknet_optim.param_groups:
        param_group["lr"] = tasknet_decay_frac * args.tasknet_lr

    policy.coeff_ent_d = args.coeff_entropy_d * descend_decay_frac
    for i in range(args.ensemble_num):
        new_log_std = args.init_log_std + (-5 - args.init_log_std) * (epoch - 1) / args.total_epochs
        policy.actor_cs[i].log_std = (torch.ones(args.svd_dim) * new_log_std).to(device)

    # ====== Agent sampling to collect transitions ======
    tasknet.eval()
    policy.train_or_eval(mode="train")
    reward_transform = Normalization(shape=1)

    for a_idx in range(args.ensemble_num):
        gnn.eval()
        batch_data_list = []
        for batch_idx, batch_data in enumerate(ens_loaders[a_idx]):
            batch_data_list.extend(batch_data.to_data_list())
            if len(batch_data_list) < args.batch_size and batch_idx < len(ens_loaders[a_idx]) - 1:
                continue
            batch_data = Batch.from_data_list(batch_data_list).to(device)
            batch_data_list = []

            nodes_per_graph = scatter(torch.ones_like(batch_data.batch), batch_data.batch, reduce="add")
            graph_num = batch_data.num_graphs
            batch_max_step = nodes_per_graph.max().item()
            state = torch.zeros(graph_num, batch_max_step, policy.max_num_nodes, args.hid_dim).to(device)
            action_d = torch.zeros(graph_num, batch_max_step, dtype=torch.long).to(device)
            action_c = torch.zeros(graph_num, batch_max_step, args.svd_dim).to(device)
            logprob_d = torch.zeros(graph_num, batch_max_step).to(device)
            logprob_c = torch.zeros(graph_num, batch_max_step).to(device)
            reward = torch.zeros(graph_num, batch_max_step + 1).to(device)
            with torch.no_grad():
                init_r = generate_reward(gnn, tasknet, batch_data)
            reward[:, 0] = init_r
            done = torch.zeros(graph_num, batch_max_step).to(device)
            valid_mask = (torch.arange(batch_max_step).expand(graph_num, -1).to(device)) < nodes_per_graph.unsqueeze(1)

            policy.prompt_slices = [torch.zeros(nodes, args.svd_dim).to(device) for nodes in nodes_per_graph]
            policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.hid_dim).to(device)
            base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
            base_idx = torch.arange(0, policy.max_num_nodes).to(device)
            policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))

            for step in range(batch_max_step):
                done[nodes_per_graph == step + 1, step] = 1.0
                s, a_d, a_c, lp_d, lp_c = policy.train_prompt(a_idx, batch_data, nodes_per_graph, step)
                state[:, step] = s
                action_d[:, step] = a_d
                action_c[:, step] = a_c
                logprob_d[:, step] = lp_d
                logprob_c[:, step] = lp_c
                r = generate_reward(gnn, tasknet, batch_data, prompt=torch.cat(policy.prompt_slices))
                reward[:, step + 1] = r

            next_state = torch.zeros_like(state)
            for i in range(graph_num):
                if nodes_per_graph[i] > 1:
                    valid_s = state[i, : nodes_per_graph[i]]
                    next_state[i, : nodes_per_graph[i] - 1] = valid_s[1:]
                    next_state[i, nodes_per_graph[i] - 1] = valid_s[-1]
                else:
                    next_state[i, 0] = state[i, 0]
            reward = reshape_reward(reward, nodes_per_graph, reward_clip=args.reward_clip)
            state = state[valid_mask]
            next_state = next_state[valid_mask]
            done = done[valid_mask]
            advantage, approx_return, approx_return0, scaled_reward = compute_adv_ret(
                args, policy.critic, state, reward, next_state, done, nodes_per_graph.tolist(), reward_transform
            )
            action_d = action_d[valid_mask]
            action_c = action_c[valid_mask]
            logprob_d = logprob_d[valid_mask]
            logprob_c = logprob_c[valid_mask]

            if state.size(0) > args.minibatch_size:
                experience = (state, action_d, logprob_d, action_c, logprob_c, advantage, approx_return)
                policy.train_policy(
                    args, a_idx, experience, nodes_per_graph, policy_optims[a_idx], approx_return0, scaled_reward, batch_idx + 1, len(ens_loaders[a_idx])
                )

    # ====== Update tasknet according to joint policy ======
    gnn.eval()
    policy.train_or_eval("eval")
    tasknet.train()
    for task_epoch in range(1, args.tasknet_epochs + 1):
        epoch_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            prompt, _, _ = attach_prompt(args, policy, batch_data, "train", gnn, tasknet, compute_reward=0, compute_prr=0)
            if getattr(args, "tasknet_train_mode", False):
                pr_loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt, policy_gnn_update=True)
            else:
                pr_loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt)
            tasknet_optim.zero_grad()
            pr_loss.backward()
            tasknet_optim.step()
            epoch_loss += pr_loss.detach().item()

    # ====== Evaluation ======
    train_acc, train_loss = evaluate_policy(args, gnn, tasknet, policy, train_loader, "train", device)
    val_acc, val_loss = evaluate_policy(args, gnn, tasknet, policy, val_loader, "val", device)
    test_acc, test_loss = evaluate_policy(args, gnn, tasknet, policy, test_loader, "test", device)
    return train_acc, val_acc, test_acc, train_loss, val_loss, test_loss
