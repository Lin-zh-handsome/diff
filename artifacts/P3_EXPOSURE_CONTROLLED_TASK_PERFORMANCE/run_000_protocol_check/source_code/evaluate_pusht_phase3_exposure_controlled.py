"""Pre-registered Phase-3 exposure-controlled Push-T rollout evaluation.

Each m condition begins from the identical saved simulator snapshot for a given
environment seed.  At each replan, a shared conditional DDIM clean-trajectory
proxy and shared forward epsilon construct a reference branch (m=0) or a
self-exposed branch (m>0).  Only task success and episode return are reported.
"""
import csv
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import time
from collections import deque, defaultdict

import click
import dill
import hydra
import matplotlib.pyplot as plt
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
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    if policy.obs_as_local_cond or policy.obs_as_global_cond:
        raise RuntimeError('Only the verified inpainting Push-T CNN is supported.')
    return cfg, policy


def policy_observation(env_observation, obs_dim):
    observation = np.asarray(env_observation, dtype=np.float32)
    if observation.shape[-1] == 2 * obs_dim:
        return observation[:obs_dim].copy()
    if observation.shape[-1] == obs_dim:
        return observation.copy()
    raise RuntimeError(f'Unexpected Push-T observation size {observation.shape[-1]}')


def build_condition(policy, history, device):
    obs = torch.as_tensor(np.stack(history), device=device, dtype=policy.dtype)[None]
    normalized_obs = policy.normalizer['obs'].normalize(obs)
    data = torch.zeros((1, policy.horizon, policy.action_dim + policy.obs_dim),
                       device=device, dtype=policy.dtype)
    mask = torch.zeros_like(data, dtype=torch.bool)
    data[:, :policy.n_obs_steps, policy.action_dim:] = normalized_obs
    mask[:, :policy.n_obs_steps, policy.action_dim:] = True
    return data, mask


def analytical_reference(scheduler, clean, noise, timestep, data, mask):
    times = torch.full((1,), timestep, device=clean.device, dtype=torch.long)
    state = scheduler.add_noise(clean, noise, times)
    state[mask] = data[mask]
    return state


def reverse_step(policy, scheduler, state, timestep, data, mask):
    state = state.clone()
    state[mask] = data[mask]
    epsilon = policy.model(state, torch.tensor(timestep, device=state.device))
    state = scheduler.step(epsilon, timestep, state, eta=0.0).prev_sample
    state[mask] = data[mask]
    return state


def sample_clean_proxy(policy, scheduler, data, mask, generator):
    """One deterministic DDIM conditional sample used as the paired clean proxy."""
    state = torch.randn(data.shape, device=data.device, dtype=data.dtype, generator=generator)
    for timestep in scheduler.timesteps:
        state = reverse_step(policy, scheduler, state, int(timestep), data, mask)
    return state


def action_plan(policy, scheduler, schedule, schedule_index, data, mask, target_timestep,
                exposure_depth, generator):
    clean_proxy = sample_clean_proxy(policy, scheduler, data, mask, generator)
    noise = torch.randn(clean_proxy.shape, device=clean_proxy.device,
                        dtype=clean_proxy.dtype, generator=generator)
    target_index = schedule_index[target_timestep]
    source_index = target_index - exposure_depth
    if source_index < 0:
        raise RuntimeError('Exposure depth has no valid DDIM source state.')
    source_timestep = schedule[source_index]
    state = analytical_reference(scheduler, clean_proxy, noise, source_timestep, data, mask)
    current_index = source_index
    # m learned transitions move the self branch to fixed target k.
    for _ in range(exposure_depth):
        state = reverse_step(policy, scheduler, state, schedule[current_index], data, mask)
        current_index += 1
    if schedule[current_index] != target_timestep:
        raise RuntimeError('Exposure branch failed to reach fixed target timestep.')
    # Both reference and self branches use the same deterministic continuation k -> 0.
    while current_index < len(schedule):
        state = reverse_step(policy, scheduler, state, schedule[current_index], data, mask)
        current_index += 1
    action = policy.normalizer['action'].unnormalize(
        state[:, policy.n_obs_steps:policy.n_obs_steps + policy.n_action_steps, :policy.action_dim]
    )[0].detach().cpu().numpy()
    return action


def create_artifact_layout(output_dir, config):
    for name in ('source_code', 'configs', 'raw_results', 'processed_results', 'figures', 'logs'):
        (output_dir / name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output_dir / 'source_code' / pathlib.Path(__file__).name)
    with (output_dir / 'configs' / 'experiment_config.json').open('w') as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    (output_dir / 'README.md').write_text(
        '# Phase 3 exposure-controlled Push-T rollout\n\n'
        'Pre-registered m conditions are compared from an identical saved simulator snapshot '
        'for each environment seed. At each replan, an m-independent conditional DDIM clean-trajectory '
        'proxy and shared forward noise construct paired reference/self branches. Reported task metrics '
        'are only success and episode return.\n',
        encoding='utf-8')
    (output_dir / 'logs' / 'run.log').write_text('started\n', encoding='utf-8')


def save_figures(summary, figures_dir):
    ms = [row['exposure_depth_m'] for row in summary]
    for metric, title, ylabel, filename in [
        ('success_rate', 'Push-T success rate', 'success rate', 'success_rate_by_m.png'),
        ('average_reward', 'Push-T average episode return', 'average reward', 'average_reward_by_m.png'),
    ]:
        figure, axis = plt.subplots(figsize=(5.5, 3.8))
        axis.plot(ms, [row[metric] for row in summary], marker='o')
        axis.set_xlabel('exposure depth m')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(ms)
        figure.tight_layout()
        figure.savefig(figures_dir / filename, dpi=180)
        plt.close(figure)


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True)
@click.option('--output_dir', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--env_seeds', default='10000,10001,10002,10003,10004,10005,10006,10007,10008,10009', show_default=True)
@click.option('--exposure_depths', default='0,1,2,4,8', show_default=True)
@click.option('--target_timestep', default=48, show_default=True, type=int)
@click.option('--num_inference_steps', default=50, show_default=True, type=int)
@click.option('--max_control_steps', default=200, show_default=True, type=int)
@click.option('--torch_seed_base', default=20260917, show_default=True, type=int)
def main(checkpoint, output_dir, device, env_seeds, exposure_depths, target_timestep,
         num_inference_steps, max_control_steps, torch_seed_base):
    """Run the pre-registered Phase-3 task-performance comparison."""
    seed_values = parse_ints(env_seeds, 'env_seeds')
    depths = parse_ints(exposure_depths, 'exposure_depths')
    if depths != [0, 1, 2, 4, 8]:
        raise click.BadParameter('Phase-3 v2.0 protocol fixes exposure_depths to 0,1,2,4,8.')
    if max_control_steps <= 0 or num_inference_steps <= 0:
        raise click.BadParameter('max_control_steps and num_inference_steps must be positive')
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_device = torch.device(device)
    cfg, policy = load_policy(checkpoint, resolved_device, output_dir)
    scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=resolved_device)
    schedule = [int(value) for value in scheduler.timesteps.tolist()]
    schedule_index = {value: index for index, value in enumerate(schedule)}
    if target_timestep not in schedule_index:
        raise click.BadParameter('target_timestep is absent from the selected DDIM schedule')
    if schedule_index[target_timestep] < max(depths):
        raise click.BadParameter('target_timestep cannot support max exposure depth')
    config = {
        'experiment_id': 'P3_EXPOSURE_CONTROLLED_TASK_PERFORMANCE',
        'env_seeds': seed_values, 'exposure_depths_m': depths,
        'target_timestep_k': target_timestep, 'num_inference_steps': num_inference_steps,
        'max_control_steps': max_control_steps, 'torch_seed_base': torch_seed_base,
        'scheduler': 'DDIMScheduler eta=0.0', 'checkpoint': str(checkpoint.resolve()),
        'task_metrics': ['success_rate', 'episode_return'],
    }
    create_artifact_layout(output_dir, config)
    records = []
    started = time.time()
    with torch.no_grad():
        for env_seed in seed_values:
            template = PushTKeypointsEnv(render_action=False)
            template.seed(env_seed)
            initial_observation = template.reset()
            initial_snapshot = template.get_state()
            initial_policy_obs = policy_observation(initial_observation, policy.obs_dim)
            for depth in depths:
                env = PushTKeypointsEnv(render_action=False)
                env.seed(env_seed)
                env.reset()
                env.set_state(initial_snapshot)
                history = deque([initial_policy_obs.copy()] * policy.n_obs_steps,
                                maxlen=policy.n_obs_steps)
                episode_return = 0.0
                success = False
                steps = 0
                decision_index = 0
                while steps < max_control_steps and not success:
                    data, mask = build_condition(policy, history, resolved_device)
                    generator_seed = torch_seed_base + env_seed * 1000 + decision_index
                    generator = torch.Generator(device=resolved_device).manual_seed(generator_seed)
                    plan = action_plan(policy, scheduler, schedule, schedule_index, data, mask,
                                       target_timestep, depth, generator)
                    for action in plan:
                        if steps >= max_control_steps:
                            break
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                        observation, reward, success, _ = env.step(action)
                        episode_return += float(reward)
                        history.append(policy_observation(observation, policy.obs_dim))
                        steps += 1
                        if success:
                            break
                    decision_index += 1
                records.append({
                    'env_seed': env_seed, 'exposure_depth_m': depth,
                    'success': int(success), 'episode_return': episode_return,
                    'control_steps': steps,
                })
                env.close()
            template.close()
    if resolved_device.type == 'cuda':
        torch.cuda.synchronize(resolved_device)
    summary = []
    reference = None
    for depth in depths:
        rows = [row for row in records if row['exposure_depth_m'] == depth]
        row = {
            'exposure_depth_m': depth, 'n_rollouts': len(rows),
            'success_rate': float(np.mean([item['success'] for item in rows])),
            'average_reward': float(np.mean([item['episode_return'] for item in rows])),
        }
        if depth == 0:
            reference = row
            row['performance_drop_success'] = 0.0
            row['performance_drop_reward'] = 0.0
        else:
            row['performance_drop_success'] = reference['success_rate'] - row['success_rate']
            row['performance_drop_reward'] = reference['average_reward'] - row['average_reward']
        summary.append(row)
    with (output_dir / 'raw_results' / 'rollouts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    manifest = {
        'experiment_id': config['experiment_id'], 'status': 'completed',
        'environment': {'hostname': socket.gethostname(), 'python': sys.version.split()[0],
                        'pytorch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(resolved_device) if resolved_device.type == 'cuda' else None},
        'code': {}, 'checkpoint': {'path': str(checkpoint.resolve()), 'epoch': '0550'},
        'diffusion': {'scheduler': 'DDIMScheduler', 'prediction_type': 'epsilon',
                      'inference_steps': num_inference_steps, 'eta': 0.0,
                      'target_k': target_timestep},
        'randomness': {'torch_seed_base': torch_seed_base, 'environment_seeds': seed_values,
                       'paired_generator_rule': 'base + env_seed*1000 + decision_index'},
        'control': {'snapshot_restore': 'identical initial snapshot restored for every m per env seed',
                    'clean_proxy_definition': 'At each replan an m-independent conditional DDIM sample '
                    'is paired with shared forward noise to construct the reference and self branches.',
                    'exposure_depths_m': depths, 'max_control_steps': max_control_steps},
        'metrics': {'success_rate': 'successful rollouts / rollouts',
                    'average_reward': 'mean episode return across rollouts',
                    'performance_drop_success': 'SR(0)-SR(m)',
                    'performance_drop_reward': 'R(0)-R(m)'},
        'artifact_layout': ['README.md', 'manifest.yaml', 'source_code/', 'configs/', 'raw_results/',
                            'processed_results/', 'figures/', 'logs/'],
        'elapsed_seconds': time.time() - started, 'summary': summary,
    }
    try:
        manifest['code']['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        manifest['code']['git_commit'] = 'unavailable'
    with (output_dir / 'processed_results' / 'summary.json').open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with (output_dir / 'manifest.yaml').open('w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    save_figures(summary, output_dir / 'figures')
    with (output_dir / 'logs' / 'run.log').open('a') as handle:
        handle.write('completed\n')
    click.echo(json.dumps({'output_dir': str(output_dir), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
