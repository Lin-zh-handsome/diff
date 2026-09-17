"""Phase 2 deterministic reverse-error propagation and reset diagnostic.

This script is read-compatible with the verified Push-T CNN checkpoint.  It
never alters policy training, inference, or weights.  A free branch and a
reset branch begin from the same analytical forward state q(x_t | x_0, epsilon).
The reset branch replaces its state after j learned DDIM reverse steps with the
same-noise analytical q-reference state, then follows the same remaining chain.
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


def parse_floats(value, name):
    result = [float(item.strip()) for item in value.split(',') if item.strip()]
    if not result:
        raise click.BadParameter(f'{name} must contain at least one number')
    return result


def masked_mse(left, right, loss_mask):
    return (left - right).square().masked_select(loss_mask).mean().item()


def reconstruct_clean(scheduler, state, epsilon_prediction, timestep):
    alpha_bar = scheduler.alphas_cumprod[timestep].to(
        device=state.device, dtype=state.dtype
    )
    return (state - (1.0 - alpha_bar).sqrt() * epsilon_prediction) / alpha_bar.sqrt()


def load_policy(checkpoint, device, output_dir):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    return cfg, policy


def make_batch(dataset, batch_size, seed):
    indexes = np.random.default_rng(seed).choice(len(dataset), size=batch_size, replace=False)
    samples = [dataset[int(index)] for index in indexes]
    return {key: torch.stack([sample[key] for sample in samples]) for key in ('obs', 'action')}


def prepare_trajectory(policy, batch, device):
    if policy.obs_as_local_cond or policy.obs_as_global_cond:
        raise RuntimeError('Phase 2 supports the verified inpainting CNN checkpoint only.')
    normalized = policy.normalizer.normalize({key: value.to(device) for key, value in batch.items()})
    clean = torch.cat([normalized['action'], normalized['obs']], dim=-1)
    condition_data = torch.zeros_like(clean)
    condition_mask = torch.zeros_like(clean, dtype=torch.bool)
    action_dim = policy.action_dim
    condition_data[:, :policy.n_obs_steps, action_dim:] = normalized['obs'][:, :policy.n_obs_steps]
    condition_mask[:, :policy.n_obs_steps, action_dim:] = True
    return clean, condition_data, condition_mask


def analytical_reference(policy, clean, noise, timestep, condition_data, condition_mask):
    timesteps = torch.full((clean.shape[0],), timestep, device=clean.device, dtype=torch.long)
    state = policy.noise_scheduler.add_noise(clean, noise, timesteps)
    state[condition_mask] = condition_data[condition_mask]
    return state


def reverse_chain(policy, scheduler, next_timestep, state, timestep, steps,
        condition_data, condition_mask):
    state = state.clone()
    for _ in range(steps):
        state[condition_mask] = condition_data[condition_mask]
        prediction = policy.model(state, torch.tensor(timestep, device=state.device))
        state = scheduler.step(prediction, timestep, state, eta=0.0).prev_sample
        state[condition_mask] = condition_data[condition_mask]
        timestep = next_timestep[timestep]
    return state, timestep


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record['source_timestep'], record['total_depth'], record['reset_depth'])].append(record)
    metrics = ('oracle_error', 'free_error', 'reset_error', 'free_minus_oracle',
        'reset_minus_oracle', 'reset_gain', 'pre_reset_state_gap', 'recovery_ratio')
    summary = []
    for key, values in sorted(groups.items()):
        source_timestep, total_depth, reset_depth = key
        row = {
            'source_timestep': source_timestep,
            'target_timestep': values[0]['target_timestep'],
            'total_depth': total_depth,
            'reset_depth': reset_depth,
            'n_seeds': len(values),
        }
        for metric in metrics:
            numbers = np.asarray([value[metric] for value in values], dtype=np.float64)
            numbers = numbers[np.isfinite(numbers)]
            row[f'{metric}_mean'] = None if len(numbers) == 0 else float(numbers.mean())
            row[f'{metric}_std'] = None if len(numbers) == 0 else float(numbers.std(ddof=0))
        summary.append(row)
    return summary


def save_curves(summary, output_dir):
    for metric, title, filename in [
        ('reset_gain', 'Reset recovery gain: E_free - E_reset', 'reset_gain_curves.png'),
        ('recovery_ratio', 'Reset recovery ratio', 'recovery_ratio_curves.png'),
    ]:
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        source_timesteps = sorted({row['source_timestep'] for row in summary}, reverse=True)
        total_depths = sorted({row['total_depth'] for row in summary})
        for source_timestep in source_timesteps:
            for total_depth in total_depths:
                rows = [row for row in summary if row['source_timestep'] == source_timestep
                    and row['total_depth'] == total_depth and row[f'{metric}_mean'] is not None]
                rows.sort(key=lambda row: row['reset_depth'])
                if rows:
                    axis.plot([row['reset_depth'] for row in rows],
                        [row[f'{metric}_mean'] for row in rows], marker='o',
                        label=f't={source_timestep}, K={total_depth}')
        axis.axhline(0.0, color='black', linewidth=0.8)
        axis.set_xlabel('reset depth j')
        axis.set_ylabel(metric)
        axis.set_title(title)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    required=True)
@click.option('--output_dir', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--batch_size', default=64, show_default=True, type=int)
@click.option('--seeds', default='0,1,2,3,4', show_default=True)
@click.option('--timesteps', default='90,50,20', show_default=True)
@click.option('--total_depths', default='4,8,16', show_default=True)
@click.option('--reset_fractions', default='0.25,0.5,0.75', show_default=True)
@click.option('--num_inference_steps', default=100, show_default=True, type=int)
def main(checkpoint, output_dir, device, batch_size, seeds, timesteps, total_depths,
        reset_fractions, num_inference_steps):
    """Run common-random-number free versus reset reverse-chain diagnostics."""
    seed_values = parse_ints(seeds, 'seeds')
    timestep_values = parse_ints(timesteps, 'timesteps')
    total_depth_values = parse_ints(total_depths, 'total_depths')
    reset_fraction_values = parse_floats(reset_fractions, 'reset_fractions')
    if batch_size <= 0 or num_inference_steps <= 0 or min(total_depth_values) <= 1:
        raise click.BadParameter('batch size, inference steps, and total depths must be positive')
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in reset_fraction_values):
        raise click.BadParameter('reset fractions must lie strictly between zero and one')
    reset_depth_map = {
        depth: sorted({max(1, min(depth - 1, round(depth * fraction)))
            for fraction in reset_fraction_values})
        for depth in total_depth_values
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_device = torch.device(device)
    cfg, policy = load_policy(checkpoint, resolved_device, output_dir)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if batch_size > len(dataset):
        raise click.BadParameter(f'batch_size exceeds dataset length {len(dataset)}')
    scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=resolved_device)
    schedule = [int(item) for item in scheduler.timesteps.tolist()]
    next_timestep = {schedule[index]: schedule[index + 1] for index in range(len(schedule) - 1)}
    if set(timestep_values) - set(schedule):
        raise click.BadParameter('one or more requested source timesteps are absent from the DDIM schedule')
    for source_timestep in timestep_values:
        current_timestep = source_timestep
        for _ in range(max(total_depth_values)):
            if current_timestep not in next_timestep:
                raise click.BadParameter(f't={source_timestep} cannot support the requested total depth')
            current_timestep = next_timestep[current_timestep]

    records = []
    started = time.time()
    with torch.no_grad():
        for seed in seed_values:
            torch.manual_seed(seed)
            clean, condition_data, condition_mask = prepare_trajectory(
                policy, make_batch(dataset, batch_size, seed), resolved_device
            )
            loss_mask = ~condition_mask
            noise = torch.randn_like(clean)
            for source_timestep in timestep_values:
                source_reference = analytical_reference(
                    policy, clean, noise, source_timestep, condition_data, condition_mask
                )
                for total_depth in total_depth_values:
                    free_state, target_timestep = reverse_chain(
                        policy, scheduler, next_timestep, source_reference, source_timestep,
                        total_depth, condition_data, condition_mask
                    )
                    reference_target = analytical_reference(
                        policy, clean, noise, target_timestep, condition_data, condition_mask
                    )
                    target_tensor = torch.tensor(target_timestep, device=resolved_device)
                    oracle_prediction = policy.model(reference_target, target_tensor)
                    free_prediction = policy.model(free_state, target_tensor)
                    oracle_error = masked_mse(
                        reconstruct_clean(scheduler, reference_target, oracle_prediction, target_timestep),
                        clean, loss_mask
                    )
                    free_error = masked_mse(
                        reconstruct_clean(scheduler, free_state, free_prediction, target_timestep),
                        clean, loss_mask
                    )
                    for reset_depth in reset_depth_map[total_depth]:
                        prefix_state, reset_timestep = reverse_chain(
                            policy, scheduler, next_timestep, source_reference, source_timestep,
                            reset_depth, condition_data, condition_mask
                        )
                        reference_reset = analytical_reference(
                            policy, clean, noise, reset_timestep, condition_data, condition_mask
                        )
                        pre_reset_gap = masked_mse(prefix_state, reference_reset, loss_mask)
                        reset_state, asserted_target = reverse_chain(
                            policy, scheduler, next_timestep, reference_reset, reset_timestep,
                            total_depth - reset_depth, condition_data, condition_mask
                        )
                        if asserted_target != target_timestep:
                            raise RuntimeError('free and reset chains reached different target timesteps')
                        reset_prediction = policy.model(reset_state, target_tensor)
                        reset_error = masked_mse(
                            reconstruct_clean(scheduler, reset_state, reset_prediction, target_timestep),
                            clean, loss_mask
                        )
                        free_minus_oracle = free_error - oracle_error
                        reset_minus_oracle = reset_error - oracle_error
                        reset_gain = free_error - reset_error
                        recovery_ratio = (reset_gain / free_minus_oracle
                            if free_minus_oracle > 1e-12 else float('nan'))
                        records.append({
                            'seed': seed,
                            'source_timestep': source_timestep,
                            'target_timestep': target_timestep,
                            'total_depth': total_depth,
                            'reset_depth': reset_depth,
                            'oracle_error': oracle_error,
                            'free_error': free_error,
                            'reset_error': reset_error,
                            'free_minus_oracle': free_minus_oracle,
                            'reset_minus_oracle': reset_minus_oracle,
                            'reset_gain': reset_gain,
                            'pre_reset_state_gap': pre_reset_gap,
                            'recovery_ratio': recovery_ratio,
                        })

    if resolved_device.type == 'cuda':
        torch.cuda.synchronize(resolved_device)
    summary = aggregate(records)
    with (output_dir / 'metrics_per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        'experiment_id': 'P2_REVERSE_RESET',
        'status': 'completed',
        'checkpoint': str(checkpoint.resolve()),
        'policy_class': type(policy).__name__,
        'device': str(resolved_device),
        'dataset_length': len(dataset),
        'batch_size': batch_size,
        'seeds': seed_values,
        'source_timesteps': timestep_values,
        'total_depths': total_depth_values,
        'reset_depths': reset_depth_map,
        'num_inference_steps': num_inference_steps,
        'scheduler': 'DDIMScheduler eta=0.0',
        'common_random_number_control': 'shared clean trajectory and forward epsilon',
        'free_branch': 'K learned DDIM reverse steps from q(x_t | x_0, epsilon)',
        'reset_branch': 'j learned steps, replace with q(x_(t-j) | x_0, epsilon), then K-j steps',
        'recovery_ratio': '(E_free - E_reset) / (E_free - E_q), reported only when denominator > 0',
        'elapsed_seconds': time.time() - started,
        'summary': summary,
    }
    with (output_dir / 'summary.json').open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    save_curves(summary, output_dir)
    click.echo(json.dumps({'output_dir': str(output_dir), 'records': len(records),
        'elapsed_seconds': manifest['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
