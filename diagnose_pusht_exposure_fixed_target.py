"""Corrected Phase-1 paired exposure diagnostic for the verified Push-T CNN.

For each fixed target DDIM timestep k and each depth m, the two branches share
one clean trajectory and forward epsilon.  The self branch starts at the DDIM
state exactly m reverse transitions before k and deterministically rolls to k;
the reference branch is analytical q(x_k | x_0, epsilon).  Primary metrics are
action-only.  Full diffusion-state values are auxiliary and never named as
primary action evidence.
"""
import csv
import json
import pathlib
import subprocess
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


def mse(left, right, mask):
    values = (left - right).square().masked_select(mask)
    if values.numel() == 0:
        raise RuntimeError('Metric mask selected no elements.')
    return float(values.mean().item())


def reconstruct_clean(scheduler, state, epsilon_prediction, timestep):
    alpha_bar = scheduler.alphas_cumprod[timestep].to(state.device, state.dtype)
    return (state - (1.0 - alpha_bar).sqrt() * epsilon_prediction) / alpha_bar.sqrt()


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
        raise RuntimeError('This corrected diagnostic supports the verified inpainting CNN only.')
    return cfg, policy


def make_batch(dataset, batch_size, seed):
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(dataset), size=batch_size, replace=False)
    samples = [dataset[int(index)] for index in indexes]
    return {key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in ('obs', 'action')}


def prepare_trajectory(policy, batch, device):
    batch = {key: value.to(device) for key, value in batch.items()}
    normalized = policy.normalizer.normalize(batch)
    clean = torch.cat([normalized['action'], normalized['obs']], dim=-1)
    data = torch.zeros_like(clean)
    condition_mask = torch.zeros_like(clean, dtype=torch.bool)
    data[:, :policy.n_obs_steps, policy.action_dim:] = normalized['obs'][:, :policy.n_obs_steps]
    condition_mask[:, :policy.n_obs_steps, policy.action_dim:] = True
    action_mask = torch.zeros_like(condition_mask, dtype=torch.bool)
    action_mask[..., :policy.action_dim] = True
    full_unconditioned_mask = ~condition_mask
    return clean, data, condition_mask, action_mask, full_unconditioned_mask


def analytical_reference(scheduler, clean, noise, timestep, data, condition_mask):
    times = torch.full((clean.shape[0],), timestep, device=clean.device, dtype=torch.long)
    state = scheduler.add_noise(clean, noise, times)
    state[condition_mask] = data[condition_mask]
    return state


def self_rollout_to_target(policy, scheduler, schedule_next, source_state, source_timestep,
                           depth, data, condition_mask):
    state = source_state.clone()
    timestep = source_timestep
    for _ in range(depth):
        state[condition_mask] = data[condition_mask]
        epsilon = policy.model(state, torch.tensor(timestep, device=state.device))
        state = scheduler.step(epsilon, timestep, state, eta=0.0).prev_sample
        state[condition_mask] = data[condition_mask]
        timestep = schedule_next[timestep]
    return state, timestep


def make_source_map(scheduler_values, targets, depths):
    index = {value: position for position, value in enumerate(scheduler_values)}
    mapping = {}
    for target in targets:
        if target not in index:
            raise click.BadParameter(f'target timestep {target} is not on the DDIM schedule')
        for depth in depths:
            source_index = index[target] - depth
            if source_index < 0:
                raise click.BadParameter(
                    f'target {target} lacks {depth} earlier DDIM reverse transitions')
            mapping[(target, depth)] = scheduler_values[source_index]
    return mapping


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['target_timestep'], row['depth'])].append(row)
    metric_names = ('G_state', 'G_pred', 'G_epsilon', 'E_q', 'E_self', 'DeltaE',
                    'G_state_X_aux', 'G_epsilon_X_aux')
    summary = []
    for (target, depth), rows in sorted(grouped.items(), reverse=True):
        aggregate_row = {'target_timestep': target, 'depth': depth, 'n_seeds': len(rows),
                         'source_timestep': rows[0]['source_timestep']}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            aggregate_row[metric + '_mean'] = float(values.mean())
            aggregate_row[metric + '_std'] = float(values.std(ddof=0))
        summary.append(aggregate_row)
    return summary


def save_heatmaps(summary, output_dir):
    metrics = ('G_state', 'G_pred', 'G_epsilon', 'E_q', 'E_self', 'DeltaE')
    targets = sorted({row['target_timestep'] for row in summary}, reverse=True)
    depths = sorted({row['depth'] for row in summary})
    for metric in metrics:
        matrix = np.full((len(targets), len(depths)), np.nan, dtype=np.float64)
        for row in summary:
            matrix[targets.index(row['target_timestep']), depths.index(row['depth'])] = row[metric + '_mean']
        figure, axis = plt.subplots(figsize=(1.5 + len(depths), 1.8 + len(targets)))
        image = axis.imshow(matrix, aspect='auto', cmap='magma')
        axis.set_xticks(range(len(depths)), labels=[str(depth) for depth in depths])
        axis.set_yticks(range(len(targets)), labels=[str(target) for target in targets])
        axis.set_xlabel('self-exposure reverse depth m')
        axis.set_ylabel('fixed target DDIM timestep k')
        axis.set_title(metric + ' (action-only)')
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, f'{value:.2e}', ha='center', va='center',
                          color='white', fontsize=8)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / ('heatmap_' + metric + '.png'), dpi=180)
        plt.close(figure)


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option('--output_dir', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--batch_size', default=64, show_default=True, type=int)
@click.option('--seeds', default='0,1,2,3,4', show_default=True,
              help='Shared diagnostic data/noise seeds.')
@click.option('--target_timesteps', default='89,49,9', show_default=True,
              help='Fixed target k values on the selected DDIM schedule (default values are for 100 steps).')
@click.option('--depths', default='1,2,4', show_default=True)
@click.option('--num_inference_steps', default=100, show_default=True, type=int)
def main(checkpoint, output_dir, device, batch_size, seeds, target_timesteps, depths,
         num_inference_steps):
    """Run corrected fixed-target, action-only Phase-1 diagnostic."""
    if batch_size <= 0 or num_inference_steps <= 0:
        raise click.BadParameter('batch_size and num_inference_steps must be positive')
    seed_values = parse_ints(seeds, 'seeds')
    target_values = parse_ints(target_timesteps, 'target_timesteps')
    depth_values = parse_ints(depths, 'depths')
    if min(depth_values) <= 0:
        raise click.BadParameter('depths must be positive')
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    cfg, policy = load_policy(checkpoint, device, output_dir)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if batch_size > len(dataset):
        raise click.BadParameter('batch_size exceeds dataset length')
    scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=device)
    values = [int(value) for value in scheduler.timesteps.tolist()]
    schedule_next = {values[i]: values[i + 1] for i in range(len(values) - 1)}
    source_map = make_source_map(values, target_values, depth_values)

    records = []
    started = time.time()
    with torch.no_grad():
        for seed in seed_values:
            torch.manual_seed(seed)
            batch = make_batch(dataset, batch_size, seed)
            clean, data, condition_mask, action_mask, full_mask = prepare_trajectory(policy, batch, device)
            noise = torch.randn_like(clean)
            for target in target_values:
                reference_state = analytical_reference(
                    scheduler, clean, noise, target, data, condition_mask)
                reference_epsilon = policy.model(reference_state, torch.tensor(target, device=device))
                reference_clean = reconstruct_clean(scheduler, reference_state, reference_epsilon, target)
                for depth in depth_values:
                    source = source_map[(target, depth)]
                    source_state = analytical_reference(
                        scheduler, clean, noise, source, data, condition_mask)
                    self_state, arrived_target = self_rollout_to_target(
                        policy, scheduler, schedule_next, source_state, source, depth,
                        data, condition_mask)
                    if arrived_target != target:
                        raise RuntimeError(
                            f'fixed-target invariant violated: requested {target}, arrived {arrived_target}')
                    self_epsilon = policy.model(self_state, torch.tensor(target, device=device))
                    self_clean = reconstruct_clean(scheduler, self_state, self_epsilon, target)
                    eq = mse(reference_clean, clean, action_mask)
                    eself = mse(self_clean, clean, action_mask)
                    records.append({
                        'data_noise_seed': seed,
                        'target_timestep': target,
                        'source_timestep': source,
                        'depth': depth,
                        'G_state': mse(self_state, reference_state, action_mask),
                        'G_pred': mse(self_clean, reference_clean, action_mask),
                        'G_epsilon': mse(self_epsilon, reference_epsilon, action_mask),
                        'E_q': eq,
                        'E_self': eself,
                        'DeltaE': eself - eq,
                        'G_state_X_aux': mse(self_state, reference_state, full_mask),
                        'G_epsilon_X_aux': mse(self_epsilon, reference_epsilon, full_mask),
                        'action_elements': int(action_mask.sum().item()),
                        'X_unconditioned_elements_aux': int(full_mask.sum().item()),
                    })
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    summary = aggregate(records)
    with (output_dir / 'metrics_per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        'experiment_id': 'P1_EXPOSURE_BIAS_FIXED_TARGET_ACTION_ONLY',
        'status': 'completed',
        'checkpoint': str(checkpoint.resolve()),
        'policy_class': type(policy).__name__,
        'device': str(device),
        'batch_size': batch_size,
        'data_noise_seeds': seed_values,
        'target_timesteps_k': target_values,
        'depths_m': depth_values,
        'num_inference_steps': num_inference_steps,
        'scheduler_timesteps': values,
        'scheduler': 'DDIMScheduler eta=0.0',
        'pairing': 'fixed target k; source is exactly m DDIM reverse transitions before k',
        'reference': 'analytical q(x_k | x_0, shared epsilon), then learned reference prediction',
        'self_exposure': 'q source followed by m deterministic learned reverse transitions to k',
        'primary_scope': 'action slice A only',
        'G_pred_definition': 'MSE between reconstructed clean action predictions',
        'G_epsilon_definition': 'MSE between epsilon predictions on action slice; auxiliary diagnostic',
        'X_auxiliary_definition': 'full unconditioned diffusion-state metrics; not primary action evidence',
        'elapsed_seconds': time.time() - started,
        'source_timestep_map': {str(key): value for key, value in source_map.items()},
        'summary': summary,
    }
    try:
        manifest['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        manifest['git_commit'] = 'unavailable'
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    save_heatmaps(summary, output_dir)
    click.echo(json.dumps({'output_dir': str(output_dir), 'records': len(records),
                           'fixed_target_timesteps': target_values}, indent=2))


if __name__ == '__main__':
    main()
