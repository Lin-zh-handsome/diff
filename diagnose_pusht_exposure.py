"""Paired-state exposure-bias diagnostic for the verified Push-T CNN policy.

The script is deliberately standalone: it loads a checkpoint and its embedded
configuration, but does not change the repository policy, training loop, or
checkpoint.  At every requested source timestep t, it compares a deterministic
DDIM self-exposure branch after m learned reverse steps against the analytical
forward-diffusion reference q(x_{t-m}|x_0, epsilon) with the same epsilon.
"""
import csv
import json
import pathlib
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


def masked_mse(left, right, loss_mask):
    difference = (left - right).square()
    return difference.masked_select(loss_mask).mean().item()


def reconstruct_clean(scheduler, state, epsilon_prediction, timestep):
    """Recover x_0 from epsilon prediction under the DDPM forward equation."""
    alpha_bar = scheduler.alphas_cumprod[timestep].to(
        device=state.device, dtype=state.dtype
    )
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
    return cfg, policy


def make_batch(dataset, batch_size, seed):
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(dataset), size=batch_size, replace=False)
    samples = [dataset[int(index)] for index in indexes]
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in ('obs', 'action')
    }


def prepare_trajectory(policy, batch, device):
    batch = {key: value.to(device) for key, value in batch.items()}
    normalized = policy.normalizer.normalize(batch)
    if policy.obs_as_local_cond or policy.obs_as_global_cond:
        raise RuntimeError(
            'This diagnostic implementation supports the evaluated CNN '
            'checkpoint only when it uses inpainting conditioning.'
        )
    trajectory = torch.cat([normalized['action'], normalized['obs']], dim=-1)
    condition_data = torch.zeros_like(trajectory)
    condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
    action_dim = policy.action_dim
    condition_data[:, :policy.n_obs_steps, action_dim:] = normalized['obs'][
        :, :policy.n_obs_steps
    ]
    condition_mask[:, :policy.n_obs_steps, action_dim:] = True
    return trajectory, condition_data, condition_mask


def analytical_reference(policy, clean, noise, timestep, condition_data, condition_mask):
    batch_size = clean.shape[0]
    timesteps = torch.full(
        (batch_size,), timestep, device=clean.device, dtype=torch.long
    )
    state = policy.noise_scheduler.add_noise(clean, noise, timesteps)
    state[condition_mask] = condition_data[condition_mask]
    return state


def reverse_self_exposure(policy, scheduler, next_timestep, start_state, start_timestep,
        depth, condition_data, condition_mask):
    state = start_state.clone()
    timestep = start_timestep
    for _ in range(depth):
        state[condition_mask] = condition_data[condition_mask]
        step_tensor = torch.tensor(timestep, device=state.device)
        prediction = policy.model(state, step_tensor)
        state = scheduler.step(
            prediction, timestep, state, eta=0.0
        ).prev_sample
        state[condition_mask] = condition_data[condition_mask]
        timestep = next_timestep[timestep]
    return state, timestep


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record['source_timestep'], record['depth'])].append(record)
    summary = []
    metric_names = ('state_gap', 'prediction_gap', 'oracle_error', 'self_error', 'delta_error')
    for (timestep, depth), values in sorted(groups.items()):
        row = {
            'source_timestep': timestep,
            'target_timestep': values[0]['target_timestep'],
            'depth': depth,
            'n_seeds': len(values),
        }
        for metric in metric_names:
            numbers = np.asarray([value[metric] for value in values], dtype=np.float64)
            row[f'{metric}_mean'] = float(numbers.mean())
            row[f'{metric}_std'] = float(numbers.std(ddof=0))
        summary.append(row)
    return summary


def save_heatmaps(summary, output_dir):
    metrics = {
        'state_gap': 'State gap',
        'prediction_gap': 'Prediction gap',
        'delta_error': 'Self minus oracle recovery error',
    }
    timesteps = sorted({row['source_timestep'] for row in summary}, reverse=True)
    depths = sorted({row['depth'] for row in summary})
    for metric, title in metrics.items():
        matrix = np.full((len(timesteps), len(depths)), np.nan, dtype=np.float64)
        for row in summary:
            matrix[timesteps.index(row['source_timestep']), depths.index(row['depth'])] = row[
                f'{metric}_mean'
            ]
        figure, axis = plt.subplots(figsize=(1.5 + len(depths), 1.8 + len(timesteps)))
        image = axis.imshow(matrix, aspect='auto', cmap='magma')
        axis.set_xticks(range(len(depths)), labels=[str(depth) for depth in depths])
        axis.set_yticks(range(len(timesteps)), labels=[str(timestep) for timestep in timesteps])
        axis.set_xlabel('reverse depth m')
        axis.set_ylabel('source timestep t')
        axis.set_title(title)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, f'{value:.3e}', ha='center', va='center',
                    color='white', fontsize=8)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / f'heatmap_{metric}.png', dpi=180)
        plt.close(figure)


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True)
@click.option('--output_dir', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--batch_size', default=64, show_default=True, type=int)
@click.option('--seeds', default='0,1,2', show_default=True)
@click.option('--timesteps', default='90,50,10', show_default=True)
@click.option('--depths', default='1,2,4', show_default=True)
@click.option('--num_inference_steps', default=100, show_default=True, type=int)
def main(checkpoint, output_dir, device, batch_size, seeds, timesteps, depths,
        num_inference_steps):
    """Run the first controlled Phase 1 exposure-bias diagnostic."""
    seed_values = parse_ints(seeds, 'seeds')
    timestep_values = parse_ints(timesteps, 'timesteps')
    depth_values = parse_ints(depths, 'depths')
    if batch_size <= 0:
        raise click.BadParameter('batch_size must be positive')
    if min(depth_values) <= 0 or min(timestep_values) <= 0:
        raise click.BadParameter('timesteps and depths must be positive')
    if num_inference_steps <= 0:
        raise click.BadParameter('num_inference_steps must be positive')

    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_device = torch.device(device)
    cfg, policy = load_policy(checkpoint, resolved_device, output_dir)
    if batch_size > 1024:
        raise click.BadParameter('batch_size is limited to 1024 for a controlled diagnostic')
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if batch_size > len(dataset):
        raise click.BadParameter(f'batch_size exceeds dataset length {len(dataset)}')

    scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=resolved_device)
    scheduler_values = [int(item) for item in scheduler.timesteps.tolist()]
    scheduler_timesteps = set(scheduler_values)
    missing_timesteps = sorted(set(timestep_values) - scheduler_timesteps)
    if missing_timesteps:
        raise RuntimeError(f'Requested timesteps are not in DDIM schedule: {missing_timesteps}')
    next_timestep = {
        timestep: scheduler_values[index + 1]
        for index, timestep in enumerate(scheduler_values[:-1])
    }
    for source_timestep in timestep_values:
        current_timestep = source_timestep
        for _ in range(max(depth_values)):
            if current_timestep not in next_timestep:
                raise click.BadParameter(
                    f'source timestep {source_timestep} cannot support all requested depths'
                )
            current_timestep = next_timestep[current_timestep]

    records = []
    started = time.time()
    with torch.no_grad():
        for seed in seed_values:
            torch.manual_seed(seed)
            batch = make_batch(dataset, batch_size, seed)
            clean, condition_data, condition_mask = prepare_trajectory(
                policy, batch, resolved_device
            )
            loss_mask = ~condition_mask
            noise = torch.randn_like(clean)
            for source_timestep in timestep_values:
                source_reference = analytical_reference(
                    policy, clean, noise, source_timestep, condition_data, condition_mask
                )
                for depth in depth_values:
                    self_state, target_timestep = reverse_self_exposure(
                        policy, scheduler, next_timestep, source_reference, source_timestep, depth,
                        condition_data, condition_mask
                    )
                    reference_state = analytical_reference(
                        policy, clean, noise, target_timestep, condition_data, condition_mask
                    )
                    target_tensor = torch.tensor(target_timestep, device=resolved_device)
                    oracle_prediction = policy.model(reference_state, target_tensor)
                    self_prediction = policy.model(self_state, target_tensor)
                    oracle_clean = reconstruct_clean(
                        scheduler, reference_state, oracle_prediction, target_timestep
                    )
                    self_clean = reconstruct_clean(
                        scheduler, self_state, self_prediction, target_timestep
                    )
                    records.append({
                        'seed': seed,
                        'source_timestep': source_timestep,
                        'target_timestep': target_timestep,
                        'depth': depth,
                        'state_gap': masked_mse(self_state, reference_state, loss_mask),
                        'prediction_gap': masked_mse(self_prediction, oracle_prediction, loss_mask),
                        'oracle_error': masked_mse(oracle_clean, clean, loss_mask),
                        'self_error': masked_mse(self_clean, clean, loss_mask),
                        'delta_error': masked_mse(self_clean, clean, loss_mask)
                            - masked_mse(oracle_clean, clean, loss_mask),
                        'unconditioned_elements': int(loss_mask.sum().item()),
                    })

    if resolved_device.type == 'cuda':
        torch.cuda.synchronize(resolved_device)
    summary = aggregate(records)
    fieldnames = list(records[0].keys())
    with (output_dir / 'metrics_per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        'experiment_id': 'P1_EXPOSURE_BIAS',
        'status': 'completed',
        'checkpoint': str(checkpoint.resolve()),
        'policy_class': type(policy).__name__,
        'device': str(resolved_device),
        'dataset_length': len(dataset),
        'batch_size': batch_size,
        'seeds': seed_values,
        'source_timesteps': timestep_values,
        'depths': depth_values,
        'num_inference_steps': num_inference_steps,
        'scheduler': 'DDIMScheduler eta=0.0',
        'shared_noise': True,
        'reference': 'analytical q(x_(t-m) | x_0, epsilon), with conditioning restored',
        'self_exposure': 'start at q(x_t | x_0, epsilon), then m DDIM learned reverse steps',
        'recovery_error': 'MSE of x_0 reconstructed from epsilon prediction, against clean x_0',
        'loss_mask': 'all trajectory elements except the two observed conditioned observation steps',
        'elapsed_seconds': time.time() - started,
        'summary': summary,
    }
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    save_heatmaps(summary, output_dir)
    click.echo(json.dumps({
        'output_dir': str(output_dir),
        'records': len(records),
        'elapsed_seconds': manifest['elapsed_seconds'],
    }, indent=2))


if __name__ == '__main__':
    main()
