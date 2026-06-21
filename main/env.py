import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import torch.optim as optim
import gymnasium as gym
import sys
import itertools
from typing import NamedTuple
import pdb



class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor

class STargmin(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, x):
        # Value-equivalence: F.one_hot(argmin) - softmax.detach() + softmax has
        # the SAME value as F.one_hot(argmin) (the softmax pair is value-zero:
        # detach() does not change a tensor's value, only its grad). The softmax
        # pair exists solely to provide the straight-through estimator gradient
        # w.r.t. x during training. When x does not require grad (the rollout /
        # SB3 path, where the action is a numpy-derived leaf), the gradient is
        # never computed anyway, so skip the two redundant softmaxes entirely.
        # Cast to float so the output dtype matches the grad path (one_hot is
        # int64; the softmax arithmetic in the grad path promotes to float).
        if not x.requires_grad:
            return F.one_hot(torch.argmin(x, dim=-1), num_classes = x.size()[-1]).float()
        return F.one_hot(torch.argmin(x, dim=-1), num_classes = x.size()[-1]) - self.softmax(-x/self.temp).detach() + self.softmax(-x/self.temp)

def allocator(action, mu, queue_counts):
    # Pure-torch allocator. Returns (allocated_work, num_allocated).
    # allocated_work: (Q, max_jobs) tensor of mu_with_grad values.
    # num_allocated: (Q,) tensor of job counts per queue.
    num_q = action.size()[-1]
    device = action.device

    adj_const = torch.where(action < 1.0, torch.tensor(1.0, device=device), action)
    mu_with_grad = mu * action / adj_const
    a = mu * action

    nz_indices = torch.nonzero(a[0] != 0)
    if nz_indices.size(0) == 0:
        return (torch.zeros(num_q, 0, device=device),
                torch.zeros(num_q, dtype=torch.int64, device=device))

    nz_s, nz_q = nz_indices[:, 0], nz_indices[:, 1]
    counts = torch.round(action[0, nz_s, nz_q]).to(torch.int64)
    total = counts.sum().item()
    if total == 0:
        return (torch.zeros(num_q, 0, device=device),
                torch.zeros(num_q, dtype=torch.int64, device=device))

    rep_s = torch.repeat_interleave(nz_s, counts)
    rep_q = torch.repeat_interleave(nz_q, counts)
    vals = mu_with_grad[0, rep_s, rep_q]

    # Sort by (queue ascending, value descending within queue) via two stable sorts
    desc_order = torch.argsort(vals, descending=True, stable=True)
    sorted_q = rep_q[desc_order]
    sorted_vals = vals[desc_order]
    q_order = torch.argsort(sorted_q, stable=True)
    sorted_q = sorted_q[q_order]
    sorted_vals = sorted_vals[q_order]

    uniq_q, uniq_counts = torch.unique_consecutive(sorted_q, return_counts=True)

    if isinstance(queue_counts, (list,)):
        q_limits = torch.tensor(queue_counts, device=device, dtype=torch.int64)
    else:
        q_limits = queue_counts.to(torch.int64)

    taken = torch.where(uniq_counts <= q_limits[uniq_q], uniq_counts, q_limits[uniq_q])
    total_kept = int(taken.sum().item())
    if total_kept == 0:
        num_allocated = torch.zeros(num_q, dtype=torch.int64, device=device)
        num_allocated[uniq_q] = taken
        return (torch.zeros(num_q, 0, device=device), num_allocated)

    seg_offsets = torch.zeros(len(uniq_q) + 1, dtype=torch.int64, device=device)
    seg_offsets[1:] = uniq_counts.cumsum(0)

    keep_list = []
    for i in range(len(uniq_q)):
        t = int(taken[i])
        if t > 0:
            start = int(seg_offsets[i])
            keep_list.append(torch.arange(start, start + t, device=device))
    keep_idx = torch.cat(keep_list)

    keep_vals = sorted_vals[keep_idx]
    keep_q = sorted_q[keep_idx]

    uniq_keep_q, uniq_keep_counts = torch.unique_consecutive(keep_q, return_counts=True)
    num_allocated = torch.zeros(num_q, dtype=torch.int64, device=device)
    num_allocated[uniq_keep_q] = uniq_keep_counts

    max_n = int(taken.max().item())
    allocated_work = torch.zeros(num_q, max_n, device=device)

    keep_offsets = torch.zeros(len(uniq_keep_q) + 1, dtype=torch.int64, device=device)
    keep_offsets[1:] = uniq_keep_counts.cumsum(0)

    group_idx = torch.searchsorted(keep_offsets, torch.arange(total_kept, device=device), right=True) - 1
    pos = torch.arange(total_kept, device=device) - keep_offsets[group_idx]
    q_ids = uniq_keep_q[group_idx]
    allocated_work[q_ids, pos] = keep_vals

    return allocated_work, num_allocated

class DiffDiscreteEventSystem(gym.Env):
    def __init__(self, network, mu, h, draw_service, draw_inter_arrivals, init_time = 0, batch = 1, queue_event_options = None,
                 straight_through_min = False,
                 queue_lim = None, temp = 1, seed = 3003,
                 max_time = 10000.0,
                 device = "cpu", f_hook = False, f_verbose = False, reset = False, use_sb = False):

        self.max_time = max_time
        self.device = device
        self.state = torch.Generator(device=self.device)
        if seed is not None:
            self.state.manual_seed(seed)
        self.network = network.repeat(batch,1,1).to(self.device)
        self.mu = mu.repeat(batch,1,1).to(self.device)
        self.q = self.network.size()[-1]
        self.s = self.network.size()[-2]
        self.h = torch.as_tensor(h, dtype=torch.float32, device=device)
        self.temp = temp
        self.st_argmin = STargmin(temp = self.temp)
        self.f_hook = f_hook
        self.f_verbose = f_verbose
        self.straight_through_min = straight_through_min
        self.batch = batch

        self.eps = 1e-8
        self.inv_eps = 1/self.eps
        self.use_sb = use_sb

        if queue_event_options is None:
            self.queue_event_options = torch.cat((F.one_hot(torch.arange(0,self.q)), -F.one_hot(torch.arange(0,self.q)))).float().to(self.device)
        else:
            self.queue_event_options = queue_event_options.float().to(self.device)

        # self.queues = init_queues.float().to(self.device)
        self.free_servers = torch.ones((self.batch, self.s)).to(self.device)
        self.cost = torch.tensor([0]).to(self.device)

        if isinstance(init_time, torch.Tensor):
            self.time_elapsed = init_time.float().to(self.device)
        else:
            self.time_elapsed = torch.tensor([0.]).to(self.device)

        self.max_steps = 10000
        self.steps = 0

        self.time_weight_queue_len = torch.zeros(self.network.size()[-1]).to(self.device)
        self.queue_len_dist = {}
        self.marg_queue_len_dist = [{} for _ in range(self.q)]
        self.terminated = False

        self.draw_service_core = draw_service
        self.draw_inter_arrivals_core = draw_inter_arrivals


        self.reset(time=self.time_elapsed, seed=seed)

    def draw_service(self, time):
        return self.draw_service_core(self, time)

    def draw_inter_arrivals(self, time):
        return self.draw_inter_arrivals_core(self, time)


    def reset(self, init_queues=None, time=None, seed = None, options:dict = None):
        #self.episode += 1
        cost = torch.tensor([0]).to(self.device)
        if time is None:
            time = torch.tensor([[0.]]).repeat(self.batch, 1).to(self.device)
        else:
            time = time.repeat(self.batch).unsqueeze(1).to(self.device)

        if init_queues is None:
            queues = torch.tensor([[0.]*self.q]).repeat(self.batch, 1).to(self.device)
        elif init_queues.size()[0] == 1:
            queues = init_queues.float().repeat(self.batch, 1).to(self.device)
        else:
            queues = init_queues.float().to(self.device)

        if seed is not None:
            self.state = torch.Generator(device=self.device)
            self.state.manual_seed(seed)

        # Tensor service times: st_data[Q, max_jobs], st_counts[Q]
        self.st_counts = (queues[0].int() if self.batch == 1 else queues[:, 0].int()).to(self.device)
        max_init = max(self.st_counts.max().item(), 1)
        self.st_data = torch.zeros((self.q, max_init), device=self.device)
        for q in range(self.q):
            c = self.st_counts[q].item()
            for j in range(c):
                self.st_data[q, j] = self.draw_service(time)[0, q]
        arrival_times = self.draw_inter_arrivals(time)

        self.obs = Obs(queues, time)
        self.env_state = EnvState(queues, time, self.st_data, arrival_times)
        self.steps = 0

        return Obs(queues, time), EnvState(queues, time, self.st_data, arrival_times)

    def step(self, action):

        # Compliance with network
        state = self.env_state
        queues, time, _, arrival_times = state
        # print(arrival_times)
        # print(queues)
        ###TODO Changed
        if self.use_sb:
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        action = action * self.network


        # action is zero if queues are zero
        #if self.f_preemptive:
        ###TODO Changed
        # action = torch.minimum(action, queues)
        action = torch.minimum(action, queues.unsqueeze(1).expand(-1,self.s,-1))

        # work is action times mu

        # allocate work to jobs
        allocated_work, num_allocated = allocator(action, self.mu, self.st_counts.to(self.device))


        if self.f_verbose:
            print(f"action:\t{action}")
            print(f"allocated work:\t{allocated_work}")

        # print(queues)
        # print(action)

        max_n = int(num_allocated.max().item())
        if max_n > 0:
            st_slice = self.st_data[:, :max_n]
            aw_slice = allocated_work[:, :max_n].clip(min=self.eps)
            mask = torch.arange(max_n, device=self.device).unsqueeze(0) < num_allocated.unsqueeze(1)
            ratio = st_slice / aw_slice
            ratio[~mask] = float('inf')
            min_eff = ratio.min(dim=1).values.unsqueeze(0)
            min_eff_service_times = torch.where(
                num_allocated.unsqueeze(0) > 0,
                min_eff,
                torch.full((1, self.q), self.inv_eps, device=self.device),
            )
        else:
            min_eff_service_times = torch.full((1, self.q), self.inv_eps, device=self.device)


        # arrival times and service times are both q vectors
        event_times = torch.cat((arrival_times, min_eff_service_times), dim=1).float()

        if self.f_verbose:
            print(f"service:\t\t{self.st_data}")
            print(f"eff service:\t\t{min_eff_service_times}")
            print(f"event times:\t\t{event_times}")
            print()

        # if a job was served, which job in which queue
        if max_n > 0:
            st_slice = self.st_data[:, :max_n]
            aw_slice = allocated_work[:, :max_n].clip(min=self.eps)
            r = st_slice / aw_slice
            r[~mask] = float('inf')
            which_job = r.argmin(dim=1)
        else:
            which_job = torch.zeros(self.q, dtype=torch.int64, device=self.device)

        which_queue = int(torch.argmin(min_eff_service_times).detach())

        if self.f_verbose:
            print(f"which_queue:\t\t{which_queue}")
            print(f"which_job:\t\t{which_job}")
            print()


        # outcome is one_hot argmin of the event times
        outcome = self.st_argmin(event_times)


        # update state based on event time

        delta_q = torch.matmul(outcome, self.queue_event_options)

        # if torch.matmul(queues[delta_q < 0], queues) == 0:
        #     print('hi')
        #     print(outcome, delta_q, self.queue_event_options)
        # compute min event
        if not self.straight_through_min:
            event_time = torch.min(event_times)
        else:
            event_time = torch.sum(event_times * outcome)

        # if self.f_verbose:
        #     print(f"outcome:\t\t{outcome}")
        #     print(f"delta_q:\t\t{delta_q}")

        # if self.f_hook:
        #     if outcome.requires_grad:
        #         event_times.register_hook(lambda grad: print(f"event_times: {grad}"))
        #         outcome.register_hook(lambda grad: print(f"outcome_grad: {grad}"))
        #         event_time.register_hook(lambda grad: print(f"event time grad: {grad}"))
        #         delta_q.register_hook(lambda grad: print(f"delta grad: {grad}"))

        # # update joint state dist: state is concatenated string
        # with torch.no_grad():
        #     state_record = self.queues.data.numpy().astype("int")
        #     joint_state_key = tuple(state_record)
        #     if joint_state_key in self.queue_len_dist.keys():
        #         self.queue_len_dist[joint_state_key] += float(event_time.data.numpy())
        #     else:
        #         self.queue_len_dist[joint_state_key] = float(event_time.data.numpy())

        #     # update marginal state dist:
        #     for qu, qu_len in enumerate(state_record):
        #         if qu_len in self.marg_queue_len_dist[qu].keys():
        #             self.marg_queue_len_dist[qu][int(qu_len)] += float(event_time.data.numpy())
        #         else:
        #             self.marg_queue_len_dist[qu][int(qu_len)] = float(event_time.data.numpy())

        # time weighted queue length
        self.time_weight_queue_len = self.time_weight_queue_len + event_time * queues

        # update time elapsed, cost, queues
        time = time + event_time
        cost = torch.matmul(event_time * queues, self.h)


        queues = F.relu(queues + delta_q)
        # pdb.set_trace()



        if self.f_verbose:
            print(f"event_time:\t\t{event_time}")
            #print(f"eff_elapsed_time:{allocated_work * event_time}")
            print()

        if max_n > 0:
            self.st_data[:, :max_n] -= event_time * allocated_work[:, :max_n]
        # update arrival times
        # if torch.min(arrival_times) == 0:
        #     assert(False)
        arrival_times = arrival_times - event_time

        if self.f_verbose:
            print(f"new service times:\t\t{self.st_data}")
            print(f"new arrival times:\t\t{arrival_times}")
            print()

        # Reset timers and add service
        # with torch.no_grad():
        if True:

            delta = delta_q.data.int()
            delta_arrived = torch.where(delta == 1, 1, 0)
            delta_left = torch.where(delta == -1, 1, 0)

            # if a new job arrives
            # print('delta_arrived', delta_arrived)
            if torch.sum(delta != 0) == 0:
                arrival_times[0,torch.argmax(outcome)] = arrival_times[0,torch.argmax(outcome)] + 1e8
            if torch.sum(delta_arrived) > 0:
                # arrival occurs
                new_arrival_times = self.draw_inter_arrivals(time)
                new_service_time = self.draw_service(time)

                # new arrival counter
                if torch.sum(delta_arrived) == 1:
                    arrival_times = arrival_times + torch.nan_to_num((new_arrival_times) * delta_arrived, nan = self.inv_eps)
                which_arrival = torch.argmax(delta_arrived)

                # service time of the new arrival (append to tensor)
                c = int(self.st_counts[which_arrival].item())
                if c >= self.st_data.size(1):
                    pad = torch.zeros((self.q, 1), device=self.device)
                    self.st_data = torch.cat([self.st_data, pad], dim=1)
                self.st_data[which_arrival, c] = new_service_time[0, which_arrival]
                self.st_counts[which_arrival] += 1

                if self.f_verbose:
                    print('Arrival!')
                    print(f"new service times:\t\t{self.st_data}")
                    print(f"new arrival times:\t\t{arrival_times}")
                    print()

            if torch.sum(delta_left) > 0:
                # remove a served job (shift left in tensor)
                j = int(which_job[which_queue].item())
                c = int(self.st_counts[which_queue].item())
                if j < c - 1:
                    self.st_data[which_queue, j:c-1] = self.st_data[which_queue, j+1:c].detach().clone()
                self.st_counts[which_queue] -= 1

                if self.f_verbose:
                    print('Service!')
                    print(f"service times:\t\t{self.st_data}")
                    print()


        next_state = EnvState(queues, time, self.st_data, arrival_times)
        obs = Obs(queues, time)

        self.env_state = next_state
        self.obs = obs

        self.steps += 1
        done = bool(self.steps >= self.max_steps)
        truncated = False
        reward = torch.clamp(-cost / 1000.0, min=-50.0, max=0.0)

        info = {"obs": obs, "state": next_state, "cost": cost, "event_time": event_time, "queues": queues}

        return queues.cpu().detach().numpy(), reward, done, truncated, info

    def get_observation(self):
        return self.queues

    def print_state(self):
        print(f"Total Cost:\t{self.cost}")
        print(f"Time Elapsed:\t{self.time_elapsed}")
        print(f"Queue Len:\t{self.queues}")
        print(f"Service times:\t{self.st_data}")
        print(f"Arrival times:\t{self.arrival_times}")
        # else:
        #     print(f"Work:\t{self.work}")


class BatchedEnv:
    """B parallel QGym envs with batched tensor operations.

    Processes one event per env per step() call, handling all B envs
    simultaneously with vectorized operations.  Replaces the serial
    DummyVecEnv Python loop with a single call.

    Uses the same allocator() and STargmin as DiffDiscreteEventSystem.
    Draw functions (draw_service / draw_inter_arrivals) must accept
    ``self`` (the BatchedEnv instance) and return ``(B, q)`` tensors.
    """

    def __init__(self, network, mu, h, draw_service, draw_inter_arrivals,
                 init_time=0, batch=50, queue_event_options=None,
                 queue_lim=None, temp=1, seed=3003, max_time=10000.0, device="cpu"):

        self.max_steps = 10000
        self.device = torch.device(device) if isinstance(device, str) else device
        self.B = batch
        self.q = network.size()[-1]
        self.s = network.size()[-2]
        self.batch = self.B  # draw_* closures reference self.batch

        self.network = network.repeat(batch, 1, 1).to(self.device)
        self.mu = mu.repeat(batch, 1, 1).to(self.device)
        self.h = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        self.temp = temp
        self.st_argmin = STargmin(temp=temp)
        self.eps = 1e-8
        self.inv_eps = 1.0 / self.eps
        self.use_sb = True

        self.state = torch.Generator(device=self.device)
        if seed is not None:
            self.state.manual_seed(seed)

        if queue_event_options is None:
            self.queue_event_options = torch.cat([
                F.one_hot(torch.arange(0, self.q)),
                -F.one_hot(torch.arange(0, self.q)),
            ]).float().to(self.device)
        else:
            self.queue_event_options = queue_event_options.float().to(self.device)

        self._draw_service = draw_service
        self._draw_inter_arrivals = draw_inter_arrivals

        self.reset()

    def draw_service(self, time):
        return self._draw_service(self, time)

    def draw_inter_arrivals(self, time):
        return self._draw_inter_arrivals(self, time)

    def reset(self):
        self.st_counts = torch.zeros(self.B, self.q, dtype=torch.int64, device=self.device)
        self.st_data = torch.zeros(self.B, self.q, 1, device=self.device)
        time = torch.zeros(self.B, 1, device=self.device)
        self.arrival_times = self.draw_inter_arrivals(time)
        self.queues = torch.zeros(self.B, self.q, device=self.device)
        self.time = time
        self.steps = np.zeros(self.B, dtype=int)
        return self.queues.detach().cpu().numpy()

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.dim() == 2 and actions.size(1) == self.s * self.q:
            actions = actions.view(-1, self.s, self.q)
        if actions.size(0) == 1 and self.B > 1:
            actions = actions.expand(self.B, -1, -1)

        actions = actions * self.network
        actions = torch.minimum(actions, self.queues.unsqueeze(1).expand(-1, self.s, -1))

        max_ub = max(int(self.st_counts.max().item()) + 1, 1)
        allocated_work = torch.zeros(self.B, self.q, max_ub, device=self.device)
        num_allocated = torch.zeros(self.B, self.q, dtype=torch.int64, device=self.device)
        for b in range(self.B):
            aw_b, na_b = allocator(actions[b:b+1], self.mu[b:b+1], self.st_counts[b])
            nb = int(na_b.max().item())
            if nb > 0:
                allocated_work[b, :, :nb] = aw_b[:, :nb]
            num_allocated[b] = na_b

        max_n = int(num_allocated.max().item())
        if max_n > 0:
            st_slice = self.st_data[:, :, :max_n]
            aw_slice = allocated_work[:, :, :max_n].clip(min=self.eps)
            mask = torch.arange(max_n, device=self.device).view(1, 1, -1) < num_allocated.unsqueeze(-1)
            ratio = st_slice / aw_slice
            ratio[~mask] = float('inf')
            min_eff = ratio.min(dim=-1).values
            which_job = ratio.argmin(dim=-1)
            min_eff_service_times = torch.where(
                num_allocated > 0, min_eff,
                torch.full((self.B, self.q), self.inv_eps, device=self.device),
            )
        else:
            min_eff_service_times = torch.full((self.B, self.q), self.inv_eps, device=self.device)
            which_job = torch.zeros(self.B, self.q, dtype=torch.int64, device=self.device)

        event_times = torch.cat([self.arrival_times, min_eff_service_times], dim=1)
        outcome = self.st_argmin(event_times)
        delta_q = torch.matmul(outcome, self.queue_event_options)
        event_time = event_times.min(dim=1).values

        self.time = self.time + event_time.unsqueeze(-1)
        cost = event_time * (self.queues @ self.h)
        self.queues = F.relu(self.queues + delta_q)
        if max_n > 0:
            self.st_data[:, :, :max_n] -= event_time.view(-1, 1, 1) * allocated_work[:, :, :max_n]
        self.arrival_times = self.arrival_times - event_time.unsqueeze(-1)

        delta = delta_q.int()
        has_arrival = (delta == 1).any(dim=1)
        if has_arrival.any():
            new_arrival = self.draw_inter_arrivals(self.time)
            new_service = self.draw_service(self.time)
            aq = outcome[:, :self.q].argmax(dim=1)
            ba = torch.where(has_arrival)[0]
            qa = aq[ba]
            ca = self.st_counts[ba, qa]
            need = (ca + 1).max().item()
            if need > self.st_data.size(-1):
                pad = torch.zeros(self.B, self.q, need - self.st_data.size(-1), device=self.device)
                self.st_data = torch.cat([self.st_data, pad], dim=-1)
            self.st_data[ba, qa, ca] = new_service[ba, qa]
            self.st_counts[ba, qa] = ca + 1
            self.arrival_times[ba, qa] = new_arrival[ba, qa]

        has_departure = (delta == -1).any(dim=1)
        if has_departure.any():
            dq = outcome[:, self.q:].argmax(dim=1)
            bd = torch.where(has_departure)[0]
            qd = dq[bd]
            jd = which_job[bd, qd]
            cd = self.st_counts[bd, qd]
            need_swap = jd < cd - 1
            if need_swap.any():
                self.st_data[bd[need_swap], qd[need_swap], jd[need_swap]] = \
                    self.st_data[bd[need_swap], qd[need_swap], cd[need_swap] - 1]
            self.st_counts[bd, qd] = cd - 1

            max_cols = int(self.st_counts.max().item()) + 1
            if max_cols * 2 < self.st_data.size(-1):
                self.st_data = self.st_data[:, :, :max_cols]

        self.steps += 1
        done = (self.steps >= self.max_steps)
        reward = torch.clamp(-cost / 1000.0, min=-50.0, max=0.0).detach().cpu().numpy()
        infos = [{} for _ in range(self.B)]
        
        if done.any():
            terminal_queues = self.queues.clone().detach().cpu().numpy()
            reset_mask = torch.tensor(done, device=self.device)
            self.steps[done] = 0
            self.time[reset_mask] = 0.0
            self.queues[reset_mask] = 0.0
            self.st_counts[reset_mask] = 0
            
            all_new_arrivals = self.draw_inter_arrivals(self.time)
            self.arrival_times[reset_mask] = all_new_arrivals[reset_mask]
            
            for b in range(self.B):
                if done[b]:
                    infos[b]["terminal_observation"] = terminal_queues[b]
                    infos[b]["TimeLimit.truncated"] = True

        return self.queues.detach().cpu().numpy(), reward, done, infos
