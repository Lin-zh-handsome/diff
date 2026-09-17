#!/usr/bin/env python3
"""Phase-3 paired Push-T control-consequence diagnostic.

For each identical simulator snapshot, construct a matched analytical-q
reference diffusion state and a deterministic DDIM self-exposure state.  Their
epsilon predictions are converted to action plans, then each plan is executed
open-loop from the *same restored snapshot*.  The clean trajectory used to
construct q is a model-sampled conditional proxy, never an expert label; all
output therefore labels this as a paired proxy-reference diagnostic.
"""
import csv
import json
import pathlib
import time
from collections import deque

import click
import dill
import hydra
import numpy as np
import torch
from diffusers import DDIMScheduler

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


def parse_ints(value, name):
    result = [int(item.strip()) for item in value.split(',') if item.strip()]
    if not result:
        raise click.BadParameter(f'{name} must contain at least one integer')
    return result


def load_policy(checkpoint, device, output_dir):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    if policy.obs_as_local_cond or policy.obs_as_global_cond:
        raise RuntimeError('Only the verified inpainting Push-T CNN is supported.')
    return cfg, policy


def build_condition(policy, obs_history, device):
    obs = torch.as_tensor(np.stack(obs_history), device=device, dtype=policy.dtype)[None]
    normalized_obs = policy.normalizer['obs'].normalize(obs)
    shape = (1, policy.horizon, policy.action_dim + policy.obs_dim)
    data = torch.zeros(shape, device=device, dtype=policy.dtype)
    mask = torch.zeros_like(data, dtype=torch.bool)
    data[:, :policy.n_obs_steps, policy.action_dim:] = normalized_obs
    mask[:, :policy.n_obs_steps, policy.action_dim:] = True
    return data, mask


def policy_observation(env_observation, obs_dim):
    """Match PushTKeypointsRunner: it feeds keypoints, not visibility masks."""
    observation = np.asarray(env_observation, dtype=np.float32)
    if observation.shape[-1] == 2 * obs_dim:
        return observation[:obs_dim].copy()
    if observation.shape[-1] == obs_dim:
        return observation.copy()
    raise RuntimeError(
        f'Push-T observation size {observation.shape[-1]} is incompatible with '
        f'policy obs_dim {obs_dim}')


def reconstruct_clean(scheduler, state, epsilon_prediction, timestep):
    alpha_bar = scheduler.alphas_cumprod[timestep].to(state.device, state.dtype)
    return (state - (1.0 - alpha_bar).sqrt() * epsilon_prediction) / alpha_bar.sqrt()


def sample_proxy(policy, data, mask, generator):
    # This is an ordinary conditional model sample used only as an x_0 proxy.
    return policy.conditional_sample(data, mask, generator=generator)


def paired_plans(policy, scheduler, next_timestep, data, mask, source_timestep,
                 depth, execution_steps, generator):
    proxy_clean = sample_proxy(policy, data, mask, generator)
    noise = torch.randn(proxy_clean.shape, device=proxy_clean.device,
                        dtype=proxy_clean.dtype, generator=generator)
    source_times = torch.full((1,), source_timestep, device=proxy_clean.device,
                              dtype=torch.long)
    reference_source = scheduler.add_noise(proxy_clean, noise, source_times)
    reference_source[mask] = data[mask]
    self_state = reference_source.clone()
    timestep = source_timestep
    for _ in range(depth):
        self_state[mask] = data[mask]
        prediction = policy.model(self_state, torch.tensor(timestep, device=self_state.device))
        self_state = scheduler.step(prediction, timestep, self_state, eta=0.0).prev_sample
        self_state[mask] = data[mask]
        timestep = next_timestep[timestep]
    reference_state = scheduler.add_noise(
        proxy_clean, noise,
        torch.full((1,), timestep, device=proxy_clean.device, dtype=torch.long))
    reference_state[mask] = data[mask]
    reference_epsilon = policy.model(reference_state, torch.tensor(timestep, device=proxy_clean.device))
    self_epsilon = policy.model(self_state, torch.tensor(timestep, device=proxy_clean.device))
    reference_clean = reconstruct_clean(scheduler, reference_state, reference_epsilon, timestep)
    self_clean = reconstruct_clean(scheduler, self_state, self_epsilon, timestep)
    start = policy.n_obs_steps
    end = start + execution_steps
    reference_plan = policy.normalizer['action'].unnormalize(
        reference_clean[:, start:end, :policy.action_dim])[0]
    self_plan = policy.normalizer['action'].unnormalize(
        self_clean[:, start:end, :policy.action_dim])[0]
    unconditioned = ~mask
    metrics = {
        'target_timestep': timestep,
        'state_gap': float((self_state[unconditioned] - reference_state[unconditioned]).square().mean()),
        'prediction_gap': float((self_epsilon[unconditioned] - reference_epsilon[unconditioned]).square().mean()),
        'proxy_reference_reconstruction_error': float(
            (reference_clean[unconditioned] - proxy_clean[unconditioned]).square().mean()),
        'self_reconstruction_error': float(
            (self_clean[unconditioned] - proxy_clean[unconditioned]).square().mean()),
    }
    return reference_plan.detach().cpu().numpy(), self_plan.detach().cpu().numpy(), metrics


def execute_plan(env, snapshot, plan):
    env.set_state(snapshot)
    rewards, infos = [], []
    done = False
    for action in plan:
        _, reward, done, info = env.step(action)
        rewards.append(float(reward))
        infos.append(info)
        if done:
            break
    final = infos[-1]
    return {
        'cumulative_reward': float(np.sum(rewards)),
        'final_reward': float(rewards[-1]),
        'success': bool(done),
        'executed_steps': len(rewards),
        'final_agent_position': np.asarray(final['pos_agent'], dtype=np.float64),
        'final_block_pose': np.asarray(final['block_pose'], dtype=np.float64),
    }


def pearson(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option('--output_dir', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--seeds', default='0,1,2,3,4', show_default=True)
@click.option('--warmup_steps', default=8, show_default=True, type=int)
@click.option('--source_timestep', default=90, show_default=True, type=int)
@click.option('--depth', default=4, show_default=True, type=int)
@click.option('--num_inference_steps', default=100, show_default=True, type=int)
@click.option('--execution_steps', default=14, show_default=True, type=int,
    help='Predicted actions to execute from each identical snapshot.')
def main(checkpoint, output_dir, device, seeds, warmup_steps, source_timestep, depth,
         num_inference_steps, execution_steps):
    """Run matched Push-T action plans from identical simulator snapshots."""
    if warmup_steps < 0 or depth <= 0:
        raise click.BadParameter('warmup_steps must be nonnegative and depth positive')
    seed_values = parse_ints(seeds, 'seeds')
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_device = torch.device(device)
    cfg, policy = load_policy(checkpoint, resolved_device, output_dir)
    max_execution_steps = policy.horizon - policy.n_obs_steps
    if execution_steps <= 0 or execution_steps > max_execution_steps:
        raise click.BadParameter(
            f'execution_steps must be in [1, {max_execution_steps}] for this policy')
    scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=resolved_device)
    values = [int(value) for value in scheduler.timesteps.tolist()]
    if source_timestep not in values:
        raise click.BadParameter('source_timestep is not on this DDIM schedule')
    next_timestep = {values[index]: values[index + 1] for index in range(len(values) - 1)}
    current = source_timestep
    for _ in range(depth):
        if current not in next_timestep:
            raise click.BadParameter('source_timestep cannot support this depth')
        current = next_timestep[current]

    records = []
    started = time.time()
    with torch.no_grad():
        for seed in seed_values:
            torch_generator = torch.Generator(device=resolved_device).manual_seed(seed)
            env = PushTKeypointsEnv(render_action=False)
            env.seed(seed)
            observation = env.reset()
            policy_obs = policy_observation(observation, policy.obs_dim)
            history = deque([policy_obs.copy()] * policy.n_obs_steps, maxlen=policy.n_obs_steps)
            # Match native runner semantics: sample a plan, then execute its
            # n_action_steps actions before replanning.  This supplies a
            # noninitial but policy-reachable snapshot without needless redraws.
            completed_warmup = 0
            while completed_warmup < warmup_steps:
                data, mask = build_condition(policy, history, resolved_device)
                warm_plan = sample_proxy(policy, data, mask, torch_generator)
                warm_actions = policy.normalizer['action'].unnormalize(
                    warm_plan[:, policy.n_obs_steps:policy.n_obs_steps + policy.n_action_steps,
                              :policy.action_dim])[0].cpu().numpy()
                for action in warm_actions:
                    if completed_warmup >= warmup_steps:
                        break
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    observation, _, done, _ = env.step(action)
                    history.append(policy_observation(observation, policy.obs_dim))
                    completed_warmup += 1
                    if done:
                        break
                if done:
                    break
            snapshot = env.get_state()
            data, mask = build_condition(policy, history, resolved_device)
            reference_plan, self_plan, metrics = paired_plans(
                policy, scheduler, next_timestep, data, mask, source_timestep, depth,
                execution_steps, torch_generator)
            unclipped_action_rmse = float(np.sqrt(np.mean((reference_plan - self_plan) ** 2)))
            reference_plan = np.clip(reference_plan, env.action_space.low, env.action_space.high)
            self_plan = np.clip(self_plan, env.action_space.low, env.action_space.high)
            paired_action_rmse = float(np.sqrt(np.mean((reference_plan - self_plan) ** 2)))
            reference_outcome = execute_plan(env, snapshot, reference_plan)
            self_outcome = execute_plan(env, snapshot, self_plan)
            record = {
                'seed': seed,
                'source_timestep': source_timestep,
                'target_timestep': metrics['target_timestep'],
                'depth': depth,
                'execution_steps': execution_steps,
                **metrics,
                'action_rmse_before_clip': unclipped_action_rmse,
                'action_rmse': paired_action_rmse,
                'reference_cumulative_reward': reference_outcome['cumulative_reward'],
                'self_cumulative_reward': self_outcome['cumulative_reward'],
                'cumulative_reward_drop': (reference_outcome['cumulative_reward']
                                           - self_outcome['cumulative_reward']),
                'reference_final_reward': reference_outcome['final_reward'],
                'self_final_reward': self_outcome['final_reward'],
                'final_reward_drop': reference_outcome['final_reward'] - self_outcome['final_reward'],
                'reference_success': int(reference_outcome['success']),
                'self_success': int(self_outcome['success']),
                'final_agent_distance': float(np.linalg.norm(
                    reference_outcome['final_agent_position'] - self_outcome['final_agent_position'])),
                'final_block_pose_distance': float(np.linalg.norm(
                    reference_outcome['final_block_pose'] - self_outcome['final_block_pose'])),
                'reference_steps': reference_outcome['executed_steps'],
                'self_steps': self_outcome['executed_steps'],
            }
            records.append(record)
            env.close()
    if resolved_device.type == 'cuda':
        torch.cuda.synchronize(resolved_device)
    with (output_dir / 'metrics_per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    reward_drop = [row['cumulative_reward_drop'] for row in records]
    pred_gap = [row['prediction_gap'] for row in records]
    manifest = {
        'experiment_id': 'P3_CONTROL_CONSEQUENCE',
        'status': 'completed',
        'checkpoint': str(checkpoint.resolve()),
        'policy_class': type(policy).__name__,
        'device': str(resolved_device),
        'seeds': seed_values,
        'warmup_steps': warmup_steps,
        'source_timestep': source_timestep,
        'depth': depth,
        'execution_steps': execution_steps,
        'target_timestep': records[0]['target_timestep'],
        'num_inference_steps': num_inference_steps,
        'scheduler': 'DDIMScheduler eta=0.0 for paired exposure branch',
        'snapshot_control': 'get_state/set_state validated before this run',
        'reference_definition': 'analytical q state from a conditional model-sampled x_0 proxy and shared epsilon',
        'self_definition': 'same q source state followed by deterministic learned DDIM reverse steps',
        'action_execution': 'open-loop predicted plans, each restored to the identical snapshot',
        'warning': 'Proxy-reference diagnostic; it is not expert-ground-truth action evaluation.',
        'elapsed_seconds': time.time() - started,
        'aggregate': {
            'n': len(records),
            'prediction_gap_mean': float(np.mean(pred_gap)),
            'action_rmse_mean': float(np.mean([row['action_rmse'] for row in records])),
            'cumulative_reward_drop_mean': float(np.mean(reward_drop)),
            'final_reward_drop_mean': float(np.mean([row['final_reward_drop'] for row in records])),
            'success_rate_reference': float(np.mean([row['reference_success'] for row in records])),
            'success_rate_self': float(np.mean([row['self_success'] for row in records])),
            'prediction_gap_vs_cumulative_reward_drop_pearson': pearson(pred_gap, reward_drop),
        },
    }
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    click.echo(json.dumps({'output_dir': str(output_dir), 'aggregate': manifest['aggregate']}, indent=2))


if __name__ == '__main__':
    main()
