"""Corrected Phase-2 action-only propagation and reset diagnostic.

This standalone diagnostic does not alter training, inference, or checkpoints.
It compares an analytical-q reference path against a learned deterministic DDIM
self path using the same clean trajectory and forward epsilon.  Per-step d_k
and R_k use action-trajectory L2 norms; reconstruction errors use action MSE.
"""
import csv
import json
import pathlib
import socket
import subprocess
import sys
import time
from collections import defaultdict

import click
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler


def parse_ints(value, name):
    result = [int(item.strip()) for item in value.split(',') if item.strip()]
    if not result:
        raise click.BadParameter(f'{name} must contain at least one integer')
    return result


def parse_floats(value, name):
    result = [float(item.strip()) for item in value.split(',') if item.strip()]
    if not result:
        raise click.BadParameter(f'{name} must contain at least one number')
    return result


def action_mse(left, right, action_dim):
    return float((left[..., :action_dim] - right[..., :action_dim]).square().mean().item())


def action_l2(left, right, action_dim):
    difference = left[..., :action_dim] - right[..., :action_dim]
    return float(torch.linalg.vector_norm(difference, dim=(-2, -1)).mean().item())


def reconstruct_clean(scheduler, state, epsilon_prediction, timestep):
    alpha_bar = scheduler.alphas_cumprod[timestep].to(state.device, state.dtype)
    return (state - (1.0 - alpha_bar).sqrt() * epsilon_prediction) / alpha_bar.sqrt()


def load_policy(checkpoint, device, output_dir):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    if policy.obs_as_local_cond or policy.obs_as_global_cond:
        raise RuntimeError('Only the verified inpainting Push-T CNN is supported.')
    return cfg, policy


def make_batch(dataset, batch_size, seed):
    indexes = np.random.default_rng(seed).choice(len(dataset), size=batch_size, replace=False)
    samples = [dataset[int(index)] for index in indexes]
    return {key: torch.stack([sample[key] for sample in samples]) for key in ('obs', 'action')}


def prepare_trajectory(policy, batch, device):
    normalized = policy.normalizer.normalize({key: value.to(device) for key, value in batch.items()})
    clean = torch.cat([normalized['action'], normalized['obs']], dim=-1)
    data = torch.zeros_like(clean)
    mask = torch.zeros_like(clean, dtype=torch.bool)
    data[:, :policy.n_obs_steps, policy.action_dim:] = normalized['obs'][:, :policy.n_obs_steps]
    mask[:, :policy.n_obs_steps, policy.action_dim:] = True
    return clean, data, mask


def reference_state(scheduler, clean, noise, timestep, data, mask):
    times = torch.full((clean.shape[0],), timestep, device=clean.device, dtype=torch.long)
    state = scheduler.add_noise(clean, noise, times)
    state[mask] = data[mask]
    return state


def learned_step(policy, scheduler, state, timestep, data, mask):
    state = state.clone()
    state[mask] = data[mask]
    epsilon = policy.model(state, torch.tensor(timestep, device=state.device))
    next_state = scheduler.step(epsilon, timestep, state, eta=0.0).prev_sample
    next_state[mask] = data[mask]
    return next_state


def aggregate(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summary = []
    for group_key, values in sorted(grouped.items()):
        row = dict(zip(keys, group_key))
        row['n_seeds'] = len(values)
        for metric in metrics:
            numbers = np.asarray([value[metric] for value in values if value[metric] is not None], dtype=np.float64)
            row[metric + '_mean'] = None if len(numbers) == 0 else float(numbers.mean())
            row[metric + '_std'] = None if len(numbers) == 0 else float(numbers.std(ddof=0))
        summary.append(row)
    return summary


def save_step_curves(step_summary, output_dir):
    for metric, title, ylabel, filename in [
        ('d_k', 'Action-only self-reference deviation', 'd_k (mean action L2)', 'd_k_curves.png'),
        ('R_k', 'Exposure Recovery Ratio', 'R_k = d_(k-1) / (d_k + epsilon)', 'R_k_curves.png'),
    ]:
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for source in sorted({row['source_timestep'] for row in step_summary}, reverse=True):
            rows = [row for row in step_summary if row['source_timestep'] == source and row[metric + '_mean'] is not None]
            rows.sort(key=lambda row: row['reverse_step'])
            if rows:
                axis.plot([row['reverse_step'] for row in rows],
                          [row[metric + '_mean'] for row in rows], marker='o', label=f'source={source}')
        if metric == 'R_k':
            axis.axhline(1.0, color='black', linewidth=0.8)
        axis.set_xlabel('learned reverse transition index')
        axis.set_ylabel(ylabel)
        axis.set_title(title + ' (action-only)')
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def save_reset_curves(reset_summary, output_dir):
    for metric, title, filename in [
        ('reset_gain', 'Reset gain: E_free - E_reset', 'reset_gain_curves.png'),
        ('reset_recovery_fraction', 'Reset recovery fraction', 'reset_recovery_fraction_curves.png'),
    ]:
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for source in sorted({row['source_timestep'] for row in reset_summary}, reverse=True):
            for total_depth in sorted({row['total_depth'] for row in reset_summary}):
                rows = [row for row in reset_summary if row['source_timestep'] == source
                        and row['total_depth'] == total_depth and row[metric + '_mean'] is not None]
                rows.sort(key=lambda row: row['reset_depth'])
                if rows:
                    axis.plot([row['reset_depth'] for row in rows],
                              [row[metric + '_mean'] for row in rows], marker='o',
                              label=f'source={source}, K={total_depth}')
        axis.axhline(0.0, color='black', linewidth=0.8)
        axis.set_xlabel('reset depth j')
        axis.set_ylabel(metric)
        axis.set_title(title + ' (action-only MSE)')
        axis.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option('--output_dir', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--batch_size', default=64, show_default=True, type=int)
@click.option('--seeds', default='0,1,2,3,4', show_default=True)
@click.option('--source_timesteps', default='95,65,35', show_default=True)
@click.option('--total_depths', default='4,8,16', show_default=True)
@click.option('--reset_fractions', default='0.25,0.5,0.75', show_default=True)
@click.option('--num_inference_steps', default=100, show_default=True, type=int)
@click.option('--ratio_epsilon', default=1e-12, show_default=True, type=float)
def main(checkpoint, output_dir, device, batch_size, seeds, source_timesteps, total_depths,
         reset_fractions, num_inference_steps, ratio_epsilon):
    """Run corrected action-only propagation and reset diagnostics."""
    seed_values = parse_ints(seeds, 'seeds')
    source_values = parse_ints(source_timesteps, 'source_timesteps')
    depth_values = parse_ints(total_depths, 'total_depths')
    reset_values = parse_floats(reset_fractions, 'reset_fractions')
    if batch_size <= 0 or min(depth_values) < 2 or ratio_epsilon <= 0:
        raise click.BadParameter('batch_size, depth, and ratio_epsilon are invalid')
    if any(value <= 0 or value >= 1 for value in reset_values):
        raise click.BadParameter('reset_fractions must be strictly between zero and one')
    reset_depths = {depth: sorted({max(1, min(depth - 1, round(depth * value)))
                                  for value in reset_values}) for depth in depth_values}
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    cfg, policy = load_policy(checkpoint, device, output_dir)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if batch_size > len(dataset):
        raise click.BadParameter('batch_size exceeds dataset length')
    scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=device)
    schedule = [int(value) for value in scheduler.timesteps.tolist()]
    next_timestep = {schedule[i]: schedule[i + 1] for i in range(len(schedule) - 1)}
    max_depth = max(depth_values)
    for source in source_values:
        current = source
        for _ in range(max_depth):
            if current not in next_timestep:
                raise click.BadParameter(f'source {source} cannot support depth {max_depth}')
            current = next_timestep[current]

    step_rows, reset_rows = [], []
    started = time.time()
    with torch.no_grad():
        for seed in seed_values:
            torch.manual_seed(seed)
            clean, data, mask = prepare_trajectory(policy, make_batch(dataset, batch_size, seed), device)
            noise = torch.randn_like(clean)
            for source in source_values:
                self_state = reference_state(scheduler, clean, noise, source, data, mask)
                trace = [(source, self_state.clone())]
                current = source
                for reverse_step in range(1, max_depth + 1):
                    reference_current = reference_state(scheduler, clean, noise, current, data, mask)
                    d_current = action_l2(self_state, reference_current, policy.action_dim)
                    next_state = learned_step(policy, scheduler, self_state, current, data, mask)
                    next_time = next_timestep[current]
                    reference_next = reference_state(scheduler, clean, noise, next_time, data, mask)
                    d_next = action_l2(next_state, reference_next, policy.action_dim)
                    ratio = d_next / (d_current + ratio_epsilon)
                    step_rows.append({
                        'data_noise_seed': seed,
                        'source_timestep': source,
                        'reverse_step': reverse_step,
                        'k_timestep': current,
                        'k_minus_1_timestep': next_time,
                        'd_k': d_current,
                        'd_k_minus_1': d_next,
                        'R_k': ratio,
                        'initial_zero_deviation': int(reverse_step == 1 and d_current <= ratio_epsilon),
                        'action_elements': int(clean.shape[0] * clean.shape[1] * policy.action_dim),
                    })
                    self_state, current = next_state, next_time
                    trace.append((current, self_state.clone()))
                for total_depth in depth_values:
                    target, free_state = trace[total_depth]
                    reference_target = reference_state(scheduler, clean, noise, target, data, mask)
                    target_tensor = torch.tensor(target, device=device)
                    reference_prediction = policy.model(reference_target, target_tensor)
                    free_prediction = policy.model(free_state, target_tensor)
                    reference_clean = reconstruct_clean(scheduler, reference_target, reference_prediction, target)
                    free_clean = reconstruct_clean(scheduler, free_state, free_prediction, target)
                    e_q = action_mse(reference_clean, clean, policy.action_dim)
                    e_free = action_mse(free_clean, clean, policy.action_dim)
                    for reset_depth in reset_depths[total_depth]:
                        reset_time, prefix_state = trace[reset_depth]
                        reference_reset = reference_state(scheduler, clean, noise, reset_time, data, mask)
                        pre_reset_d = action_l2(prefix_state, reference_reset, policy.action_dim)
                        reset_state = reference_reset.clone()
                        replay_time = reset_time
                        for _ in range(total_depth - reset_depth):
                            reset_state = learned_step(policy, scheduler, reset_state, replay_time, data, mask)
                            replay_time = next_timestep[replay_time]
                        if replay_time != target:
                            raise RuntimeError('reset and free branches reached different target timesteps')
                        reset_prediction = policy.model(reset_state, target_tensor)
                        reset_clean = reconstruct_clean(scheduler, reset_state, reset_prediction, target)
                        e_reset = action_mse(reset_clean, clean, policy.action_dim)
                        excess_error = e_free - e_q
                        reset_gain = e_free - e_reset
                        fraction = reset_gain / excess_error if excess_error > ratio_epsilon else None
                        reset_rows.append({
                            'data_noise_seed': seed,
                            'source_timestep': source,
                            'target_timestep': target,
                            'total_depth': total_depth,
                            'reset_depth': reset_depth,
                            'reset_timestep': reset_time,
                            'E_q': e_q,
                            'E_free': e_free,
                            'E_reset': e_reset,
                            'reset_gain': reset_gain,
                            'reset_recovery_fraction': fraction,
                            'pre_reset_d_action_l2': pre_reset_d,
                            'action_elements': int(clean.shape[0] * clean.shape[1] * policy.action_dim),
                        })
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    step_summary = aggregate(step_rows, ('source_timestep', 'reverse_step', 'k_timestep', 'k_minus_1_timestep'),
                             ('d_k', 'd_k_minus_1', 'R_k'))
    reset_summary = aggregate(reset_rows, ('source_timestep', 'target_timestep', 'total_depth', 'reset_depth'),
                              ('E_q', 'E_free', 'E_reset', 'reset_gain', 'reset_recovery_fraction', 'pre_reset_d_action_l2'))
    with (output_dir / 'per_step_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0]))
        writer.writeheader(); writer.writerows(step_rows)
    with (output_dir / 'reset_metrics_per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reset_rows[0]))
        writer.writeheader(); writer.writerows(reset_rows)
    manifest = {
        'experiment_id': 'P2_ACTION_ONLY_PROPAGATION_AND_RESET',
        'status': 'completed', 'checkpoint': str(checkpoint.resolve()),
        'policy_class': type(policy).__name__, 'prediction_type': 'epsilon',
        'host': socket.gethostname(), 'python': sys.version.split()[0],
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'device': str(device), 'dataset': str(cfg.task.dataset.zarr_path),
        'dataset_length': len(dataset), 'batch_size': batch_size,
        'data_noise_seeds': seed_values, 'scheduler': 'DDIMScheduler eta=0.0',
        'num_inference_steps': num_inference_steps, 'scheduler_timesteps': schedule,
        'source_timesteps': source_values, 'total_depths': depth_values,
        'reset_depths': reset_depths, 'shared_forward_epsilon': True,
        'deterministic': True, 'action_scope': 'all action dimensions only',
        'd_k_definition': 'mean over batch of action-trajectory L2 norms',
        'R_k_definition': 'd_(k-1) / (d_k + ratio_epsilon), action-only L2 ratio',
        'ratio_epsilon': ratio_epsilon,
        'E_definition': 'action-only reconstructed-clean-action MSE to clean action',
        'reset_recovery_fraction_definition': '(E_free-E_reset)/(E_free-E_q); distinct from R_k',
        'reference_branch': 'analytical q(x_k | x_0, shared epsilon)',
        'self_branch': 'learned deterministic DDIM reverse path from same source state',
        'reset_branch': 'replace learned prefix with same-noise analytical reference then deterministic replay',
        'artifact_paths': ['per_step_metrics.csv', 'reset_metrics_per_seed.csv', 'summary.json',
                           'd_k_curves.png', 'R_k_curves.png', 'reset_gain_curves.png',
                           'reset_recovery_fraction_curves.png'],
        'elapsed_seconds': time.time() - started,
        'step_summary': step_summary, 'reset_summary': reset_summary,
    }
    try:
        manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        manifest['git_commit'] = 'unavailable'
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    save_step_curves(step_summary, output_dir)
    save_reset_curves(reset_summary, output_dir)
    click.echo(json.dumps({'output_dir': str(output_dir), 'step_records': len(step_rows),
                           'reset_records': len(reset_rows)}, indent=2))


if __name__ == '__main__':
    main()
