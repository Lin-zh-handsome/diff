#!/usr/bin/env python3
"""Replay test for paired Push-T simulator snapshots.

This is a Phase-3 preflight: it never calls the policy.  It proves that an
identical action sequence from a restored snapshot yields the same observation,
reward, terminal flag, info fields and final physics state.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def run_actions(env, actions):
    records = []
    for action in actions:
        obs, reward, done, info = env.step(np.asarray(action, dtype=np.float64))
        records.append({
            'observation': np.asarray(obs).copy(),
            'reward': float(reward),
            'done': bool(done),
            'info': {key: np.asarray(value).copy() if isinstance(value, np.ndarray)
                     else value for key, value in info.items()},
        })
    return records, env.get_state()


def max_abs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def compare_records(first, second):
    errors = []
    for index, (left, right) in enumerate(zip(first, second)):
        errors.append({
            'step': index,
            'observation_max_abs': max_abs(left['observation'], right['observation']),
            'reward_abs': abs(left['reward'] - right['reward']),
            'done_equal': left['done'] == right['done'],
            'info_max_abs': {key: max_abs(left['info'][key], right['info'][key])
                             for key in left['info']},
        })
    return errors


def state_errors(first, second):
    output = {}
    for body_name in ('agent', 'block'):
        output[body_name] = {
            key: max_abs(first[body_name][key], second[body_name][key])
            for key in first[body_name]
        }
    output['goal_pose'] = max_abs(first['goal_pose'], second['goal_pose'])
    output['n_contact_points_equal'] = (
        first['n_contact_points'] == second['n_contact_points'])
    output['latest_action'] = max_abs(first['latest_action'], second['latest_action'])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--atol', type=float, default=1e-10)
    args = parser.parse_args()

    env = PushTKeypointsEnv(keypoint_visible_rate=0.65, render_action=False)
    env.seed(args.seed)
    env.reset()
    # Produce non-zero velocity/contact history before taking the snapshot.
    for action in ((420, 120), (310, 210), (260, 280)):
        env.step(np.asarray(action, dtype=np.float64))
    snapshot = env.get_state()
    actions = ((200, 280), (160, 340), (340, 230), (400, 180), (290, 250))
    branch_a, final_a = run_actions(env, actions)
    env.set_state(snapshot)
    branch_b, final_b = run_actions(env, actions)

    record_errors = compare_records(branch_a, branch_b)
    final_errors = state_errors(final_a, final_b)
    numeric_errors = []
    for entry in record_errors:
        numeric_errors.extend([entry['observation_max_abs'], entry['reward_abs']])
        numeric_errors.extend(entry['info_max_abs'].values())
    for body in ('agent', 'block'):
        numeric_errors.extend(final_errors[body].values())
    numeric_errors.extend([final_errors['goal_pose'], final_errors['latest_action']])
    passed = (max(numeric_errors, default=0.0) <= args.atol
              and all(entry['done_equal'] for entry in record_errors)
              and final_errors['n_contact_points_equal'])

    result = {
        'test': 'PushT identical-snapshot replay',
        'seed': args.seed,
        'atol': args.atol,
        'actions': [list(action) for action in actions],
        'passed': passed,
        'record_errors': record_errors,
        'final_state_errors': final_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(result), indent=2), encoding='utf-8')
    env.close()
    if not passed:
        raise SystemExit('snapshot replay mismatch; see ' + str(args.output))


if __name__ == '__main__':
    main()
