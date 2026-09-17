"""Short real-batch Phase-3 Self-Exposure loss/backward test for Push-T."""
import json
import pathlib

import click
import dill
import hydra
import torch


def make_batch(dataset, batch_size, device):
    samples = [dataset[index] for index in range(batch_size)]
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0).to(device)
        for key in ('obs', 'action')
    }


@click.command()
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False,
                                               path_type=pathlib.Path), required=True)
@click.option('--output_path', type=click.Path(path_type=pathlib.Path), required=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--batch_size', default=4, show_default=True, type=int)
def main(checkpoint, output_path, device, batch_size):
    """Run one real Push-T minibatch through detached self-exposure and backward."""
    device = torch.device(device)
    payload = torch.load(checkpoint.open('rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(output_path.parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model.to(device)
    policy.train()
    policy.self_exposure_enabled = True
    policy.self_exposure_weight = 0.1
    policy.self_exposure_depth = 1
    policy.self_exposure_num_inference_steps = 100

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = make_batch(dataset, batch_size, device)
    policy.zero_grad(set_to_none=True)
    total_loss = policy.compute_loss(batch)
    total_loss.backward()
    gradient_is_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in policy.parameters()
    )
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result = {
        'checkpoint': str(checkpoint.resolve()),
        'batch_size': batch_size,
        'losses': policy.get_last_loss_components(),
        'self_exposure': policy.get_last_self_exposure_info(),
        'total_loss_backward_completed': True,
        'all_present_gradients_finite': bool(gradient_is_finite),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    click.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
