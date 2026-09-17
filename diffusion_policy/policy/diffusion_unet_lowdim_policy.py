from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            self_exposure_enabled=False,
            self_exposure_weight=0.0,
            self_exposure_depth=1,
            self_exposure_num_inference_steps=100,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.self_exposure_enabled = bool(self_exposure_enabled)
        self.self_exposure_weight = float(self_exposure_weight)
        self.self_exposure_depth = int(self_exposure_depth)
        self.self_exposure_num_inference_steps = int(self_exposure_num_inference_steps)
        if self.self_exposure_depth != 1:
            raise ValueError('Self-Exposure v1 fixes depth to m=1.')
        if self.self_exposure_weight < 0:
            raise ValueError('self_exposure_weight must be non-negative.')
        if self.self_exposure_num_inference_steps <= 1:
            raise ValueError('self_exposure_num_inference_steps must exceed one.')
        self.self_exposure_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        self._last_loss_components = None
        self._last_self_exposure_info = None
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_last_loss_components(self):
        if self._last_loss_components is None:
            return None
        return dict(self._last_loss_components)

    def get_last_self_exposure_info(self):
        if self._last_self_exposure_info is None:
            return None
        return dict(self._last_self_exposure_info)

    def _compute_self_exposure_loss(self, trajectory, condition_mask,
                                    local_cond, global_cond):
        """Build one detached DDIM self-exposure state and supervise actions only."""
        batch_size = trajectory.shape[0]
        scheduler = self.self_exposure_scheduler
        scheduler.set_timesteps(
            self.self_exposure_num_inference_steps, device=trajectory.device)
        timesteps = scheduler.timesteps
        # DDIM timesteps are ordered from high to low noise. A valid target has
        # one preceding entry, which is its actual upstream reverse predecessor.
        target_index = torch.randint(
            1, len(timesteps), (1,), device=trajectory.device).item()
        source_timestep = int(timesteps[target_index - 1].item())
        target_timestep = int(timesteps[target_index].item())
        source_times = torch.full(
            (batch_size,), source_timestep,
            device=trajectory.device, dtype=torch.long)

        with torch.no_grad():
            exposure_noise = torch.randn_like(trajectory)
            source_state = scheduler.add_noise(trajectory, exposure_noise, source_times)
            source_state[condition_mask] = trajectory[condition_mask]
            source_epsilon = self.model(
                source_state, source_times,
                local_cond=local_cond, global_cond=global_cond)
            self_state = scheduler.step(
                source_epsilon, source_timestep, source_state, eta=0.0
            ).prev_sample
            self_state[condition_mask] = trajectory[condition_mask]

        target_times = torch.full(
            (batch_size,), target_timestep,
            device=trajectory.device, dtype=torch.long)
        self_epsilon = self.model(
            self_state, target_times,
            local_cond=local_cond, global_cond=global_cond)
        alpha_bar = scheduler.alphas_cumprod[target_timestep].to(
            device=trajectory.device, dtype=trajectory.dtype)
        predicted_clean_action = (
            self_state[..., :self.action_dim]
            - torch.sqrt(1.0 - alpha_bar) * self_epsilon[..., :self.action_dim]
        ) / torch.sqrt(alpha_bar)
        loss = F.mse_loss(predicted_clean_action, trajectory[..., :self.action_dim])
        self._last_self_exposure_info = {
            'source_timestep': source_timestep,
            'target_timestep': target_timestep,
            'eta': 0.0,
            'depth': 1,
        }
        return loss

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        l_dp = loss.mean()
        if not self.self_exposure_enabled:
            self._last_loss_components = {
                'l_dp': float(l_dp.detach().item()),
                'l_se': 0.0,
                'l_total': float(l_dp.detach().item()),
            }
            self._last_self_exposure_info = None
            return l_dp

        l_se = self._compute_self_exposure_loss(
            trajectory, condition_mask, local_cond, global_cond)
        total_loss = l_dp + self.self_exposure_weight * l_se
        self._last_loss_components = {
            'l_dp': float(l_dp.detach().item()),
            'l_se': float(l_se.detach().item()),
            'l_total': float(total_loss.detach().item()),
        }
        return total_loss
