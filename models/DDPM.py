import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
            )
    assert log_probs.shape == x.shape
    return log_probs




class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_type="l2",  # l2,l1 hybrid
            loss_weight='none',  # prop t / uniform / None
            noise="gauss",  # gauss / perlin / simplex
            ):
        super().__init__()

        if noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)

        self.img_size = img_size
        self.img_channels = img_channels
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        self.gauss_blur = T.GaussianBlur(kernel_size=31, sigma=3)
        


    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))

        :param model:
        :param x_t:
        :param t:
        :return:
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
                pred_x_0, x_t, t
                )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
            }

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        # noise = torch.randn_like(x_t)
        if denoise_fn == "gauss":
            noise = torch.randn_like(x_t) 
        else:
            noise = denoise_fn(x_t, t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":

            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                # noise = torch.randn_like(x)
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q(
                    x, t_tensor,
                    self.noise_fn(x, t_tensor).float()
                    )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq

    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, output["mean"], log_scales=0.5 * output["log_variance"]
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t):

        noise = self.noise_fn(x_0, t).float()
        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)
        loss = {}
        if self.loss_type == "l1":
            loss["loss"] = mean_flat((estimate_noise - noise).abs())
        elif self.loss_type == "l2":
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        elif self.loss_type == "hybrid":
            # add vlb term
            loss["vlb"] = self.calc_vlb_xt(model, x_0, x_t, t, estimate_noise)["output"]
            loss["loss"] = loss["vlb"] + mean_flat((estimate_noise - noise).square())
        else:
            loss["loss"] = mean_flat((estimate_noise - noise).square())
        return loss, x_t, estimate_noise

    @staticmethod
    def _get_arg_value(args, key, default):
        if hasattr(args, "get"):
            value = args.get(key, default)
        else:
            try:
                value = args[key]
            except Exception:
                value = default
        if value == "" or value is None:
            return default
        return value

    @classmethod
    def _get_arg_bool(cls, args, key, default=False):
        value = cls._get_arg_value(args, key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def _get_arg_float(cls, args, key, default=0.0):
        value = cls._get_arg_value(args, key, default)
        return float(value)

    def _build_timestep_segment_weight(self, normal_t, args, reference_tensor):
        """
        Piecewise guidance schedule over normal_t:
        early (t <= t1), middle (t1 < t < t2), late (t >= t2).
        """
        if not self._get_arg_bool(args, "guidance_timestep_schedule_enable", False):
            return None

        t1 = int(self._get_arg_float(args, "guidance_timestep_schedule_t1", 100))
        t2 = int(self._get_arg_float(args, "guidance_timestep_schedule_t2", 220))
        if t2 < t1:
            t1, t2 = t2, t1

        w_early = self._get_arg_float(args, "guidance_timestep_schedule_w_early", 0.9)
        w_mid = self._get_arg_float(args, "guidance_timestep_schedule_w_mid", 1.1)
        w_late = self._get_arg_float(args, "guidance_timestep_schedule_w_late", 0.9)

        segment_weight = torch.full(normal_t.shape, w_mid, device=normal_t.device, dtype=torch.float32)
        early_weight = torch.full_like(segment_weight, w_early)
        late_weight = torch.full_like(segment_weight, w_late)

        segment_weight = torch.where(normal_t <= t1, early_weight, segment_weight)
        segment_weight = torch.where(normal_t >= t2, late_weight, segment_weight)

        if self._get_arg_bool(args, "guidance_timestep_schedule_detach_weight", True):
            segment_weight = segment_weight.detach()

        view_shape = [segment_weight.shape[0]] + [1] * (reference_tensor.ndim - 1)
        return segment_weight.view(*view_shape).to(dtype=reference_tensor.dtype, device=reference_tensor.device)

    def _build_guidance_scale(self, normal_t, args, reference_tensor, base_scale=1.0):
        base_scale = float(base_scale)
        segment_weight = self._build_timestep_segment_weight(normal_t, args, reference_tensor)
        if segment_weight is None:
            return base_scale
        return segment_weight * base_scale

    def _compute_irf_guidance_weight(self, residual, normal_t, args, anomaly_label=None):
        """
        Build an IRF-based sample-wise guidance weight from whitened residuals.
        """
        norm_eps = self._get_arg_float(args, "single_path_irf_eps", 1e-6)
        residual_norm = extract(
            self.sqrt_one_minus_alphas_cumprod,
            normal_t,
            residual.shape,
            residual.device,
        )
        irf_residual = residual / (residual_norm + norm_eps)

        irf_clip = self._get_arg_float(args, "single_path_irf_clip", 0.0)
        if irf_clip > 0:
            irf_residual = torch.clamp(irf_residual, -irf_clip, irf_clip)

        score_mode = str(self._get_arg_value(args, "single_path_irf_score_mode", "l2")).strip().lower()
        if score_mode == "l1":
            irf_score = mean_flat(irf_residual.abs())
        elif score_mode in {"gaussian", "nll", "gaussian_nll"}:
            ref_irf = irf_residual.detach()
            if anomaly_label is not None:
                normal_mask = anomaly_label == 0
                if normal_mask.ndim == 0:
                    normal_mask = normal_mask.unsqueeze(0)
                if normal_mask.any():
                    ref_irf = ref_irf[normal_mask]

            ref_mean = ref_irf.mean()
            ref_var = ref_irf.var(unbiased=False)
            nll = 0.5 * (
                (irf_residual - ref_mean).square() / (ref_var + norm_eps)
                + torch.log(ref_var + norm_eps)
            )
            irf_score = mean_flat(nll)
        else:
            irf_score = mean_flat(irf_residual.square())

        if self._get_arg_bool(args, "single_path_irf_adaptive_tau", False):
            ref_score = irf_score.detach()
            if anomaly_label is not None:
                normal_mask = anomaly_label == 0
                if normal_mask.ndim == 0:
                    normal_mask = normal_mask.unsqueeze(0)
                if normal_mask.any():
                    ref_score = ref_score[normal_mask]

            tau = ref_score.mean()
            if ref_score.numel() > 1:
                tau = tau + self._get_arg_float(args, "single_path_irf_tau_std_scale", 1.0) * ref_score.std(
                    unbiased=False
                )
        else:
            tau = torch.tensor(
                self._get_arg_float(args, "single_path_irf_tau", 1.0),
                device=residual.device,
                dtype=irf_score.dtype,
            )

        sigmoid_beta = self._get_arg_float(args, "single_path_irf_sigmoid_beta", 4.0)
        gate = torch.sigmoid((irf_score - tau) * sigmoid_beta)

        min_w = self._get_arg_float(args, "single_path_irf_min_weight", 0.5)
        max_w = self._get_arg_float(args, "single_path_irf_max_weight", 1.5)
        if max_w < min_w:
            min_w, max_w = max_w, min_w

        irf_weight = min_w + (max_w - min_w) * gate
        if self._get_arg_bool(args, "single_path_irf_detach_weight", True):
            irf_weight = irf_weight.detach()

        view_shape = [irf_weight.shape[0]] + [1] * (residual.ndim - 1)
        irf_weight = irf_weight.view(*view_shape).to(dtype=residual.dtype)
        return irf_residual, irf_weight

    def _apply_irf_guidance_weight(self, residual, normal_t, args, anomaly_label=None):
        if not self._get_arg_bool(args, "single_path_irf_weight_enable", False):
            return residual
        _irf_residual, irf_weight = self._compute_irf_guidance_weight(
            residual,
            normal_t,
            args,
            anomaly_label=anomaly_label,
        )
        return residual * irf_weight

    def _build_single_path_proxy_variant(
            self,
            model,
            x_normal_t,
            normal_t,
            estimate_noise_normal,
            mode="self",
            delta_t=200,
            proxy_mix=1.0,
            detach_high=False,
    ):
        """
        Build a proxy for the missing noisier branch in single-path mode.

        Modes:
        - self (legacy): reproject from normal branch only.
        - pseudo_noisier: create a pseudo high-t state and project it back.
        """
        pred_x_0_normal = self.predict_x_0_from_eps(
            x_normal_t, normal_t, estimate_noise_normal
        ).clamp(-1, 1)
        pred_x_t_self = self.sample_q(pred_x_0_normal, normal_t, estimate_noise_normal)

        mode = str(mode).strip().lower()

        pred_x_t_proxy = pred_x_t_self
        proxy_state_t = pred_x_t_self
        pred_x_0_proxy = pred_x_0_normal

        if mode in {"pseudo_noisier", "pseudo-high", "pseudo_high", "v2"}:
            delta_t = max(1, int(delta_t))
            proxy_mix = float(max(0.0, min(1.0, proxy_mix)))

            high_t = torch.clamp(normal_t + delta_t, max=self.num_timesteps - 1)
            x_pseudo_noisier_t = self.sample_q(pred_x_0_normal, high_t, estimate_noise_normal)
            if detach_high:
                x_pseudo_noisier_t = x_pseudo_noisier_t.detach()

            estimate_noise_high = model(x_pseudo_noisier_t, high_t)
            pred_x_0_high = self.predict_x_0_from_eps(
                x_pseudo_noisier_t, high_t, estimate_noise_high
            ).clamp(-1, 1)
            pred_x_t_high = self.sample_q(pred_x_0_high, normal_t, estimate_noise_normal)

            pred_x_t_proxy = (1.0 - proxy_mix) * pred_x_t_self + proxy_mix * pred_x_t_high
            proxy_state_t = x_pseudo_noisier_t
            pred_x_0_proxy = pred_x_0_high

        return pred_x_0_normal, pred_x_t_proxy, proxy_state_t, pred_x_0_proxy

    def _build_single_path_proxy(self, model, x_normal_t, normal_t, estimate_noise_normal, args):
        mode = str(self._get_arg_value(args, "single_path_proxy_mode", "self")).strip().lower()
        delta_t = self._get_arg_float(args, "single_path_delta_t", 200)
        proxy_mix = self._get_arg_float(args, "single_path_proxy_mix", 1.0)
        detach_high = self._get_arg_bool(args, "single_path_proxy_detach_high", False)
        return self._build_single_path_proxy_variant(
            model,
            x_normal_t,
            normal_t,
            estimate_noise_normal,
            mode=mode,
            delta_t=delta_t,
            proxy_mix=proxy_mix,
            detach_high=detach_high,
        )

    def _apply_consistency_guidance_weight(
            self,
            model,
            residual,
            x_normal_t,
            normal_t,
            estimate_noise_normal,
            args,
    ):
        """
        Native dual-proxy consistency weighting:
        compare primary residual with a secondary proxy residual and scale guidance
        by their sample-wise cosine consistency.
        """
        if not self._get_arg_bool(args, "single_path_consistency_enable", False):
            return residual

        second_mode = str(
            self._get_arg_value(args, "single_path_consistency_proxy_mode", "pseudo_noisier")
        ).strip().lower()
        second_delta_t = self._get_arg_float(args, "single_path_consistency_delta_t", 160)
        second_proxy_mix = self._get_arg_float(args, "single_path_consistency_proxy_mix", 1.0)
        second_detach_high = self._get_arg_bool(args, "single_path_consistency_detach_high", True)

        _, pred_x_t_proxy_2, _, _ = self._build_single_path_proxy_variant(
            model,
            x_normal_t,
            normal_t,
            estimate_noise_normal,
            mode=second_mode,
            delta_t=second_delta_t,
            proxy_mix=second_proxy_mix,
            detach_high=second_detach_high,
        )

        second_scale = self._get_arg_float(args, "single_path_consistency_residual_scale", 1.0)
        residual_2 = (pred_x_t_proxy_2 - x_normal_t) * second_scale

        cos_eps = self._get_arg_float(args, "single_path_consistency_cosine_eps", 1e-6)
        if self._get_arg_bool(args, "single_path_consistency_normalize", True):
            norm_eps = self._get_arg_float(args, "single_path_consistency_norm_eps", 1e-6)
            residual_norm = extract(
                self.sqrt_one_minus_alphas_cumprod,
                normal_t,
                residual.shape,
                residual.device,
            )
            residual_1_view = residual / (residual_norm + norm_eps)
            residual_2_view = residual_2 / (residual_norm + norm_eps)
        else:
            residual_1_view = residual
            residual_2_view = residual_2

        residual_1_flat = residual_1_view.flatten(start_dim=1)
        residual_2_flat = residual_2_view.flatten(start_dim=1)
        consistency_cos = F.cosine_similarity(
            residual_1_flat,
            residual_2_flat,
            dim=1,
            eps=cos_eps,
        )
        consistency_gate = (consistency_cos + 1.0) * 0.5

        gate_tau = self._get_arg_float(args, "single_path_consistency_tau", 0.5)
        gate_beta = self._get_arg_float(args, "single_path_consistency_beta", 8.0)
        consistency_gate = torch.sigmoid((consistency_gate - gate_tau) * gate_beta)

        min_w = self._get_arg_float(args, "single_path_consistency_min_weight", 0.5)
        max_w = self._get_arg_float(args, "single_path_consistency_max_weight", 1.5)
        if max_w < min_w:
            min_w, max_w = max_w, min_w

        consistency_weight = min_w + (max_w - min_w) * consistency_gate
        if self._get_arg_bool(args, "single_path_consistency_detach_weight", True):
            consistency_weight = consistency_weight.detach()

        view_shape = [consistency_weight.shape[0]] + [1] * (residual.ndim - 1)
        consistency_weight = consistency_weight.view(*view_shape).to(dtype=residual.dtype)
        return residual * consistency_weight

    def _sample_single_path_distill_noisier_t(self, normal_t, args):
        if self._get_arg_bool(args, "single_path_distill_use_delta_t", False):
            delta_t = max(1, int(self._get_arg_float(args, "single_path_distill_delta_t", 200)))
            return torch.clamp(normal_t + delta_t, max=self.num_timesteps - 1)

        min_t = int(self._get_arg_value(args, "single_path_distill_min_t", args["less_t_range"]))
        min_t = max(0, min_t)
        if min_t >= self.num_timesteps:
            min_t = self.num_timesteps - 1

        if min_t >= self.num_timesteps - 1:
            return torch.full_like(normal_t, self.num_timesteps - 1)

        return torch.randint(min_t, self.num_timesteps, normal_t.shape, device=normal_t.device)

    def _calc_single_path_residual_distill_loss(
            self,
            model,
            x_0,
            x_normal_t,
            normal_t,
            estimate_noise_normal,
            pred_x_t_proxy,
            anomaly_label,
            args,
    ):
        if not self._get_arg_bool(args, "single_path_residual_distill", False):
            return x_normal_t.new_zeros(())

        distill_weight = self._get_arg_float(args, "single_path_distill_weight", 0.0)
        if distill_weight <= 0:
            return x_normal_t.new_zeros(())

        distill_prob = self._get_arg_float(args, "single_path_distill_prob", 1.0)
        distill_prob = float(max(0.0, min(1.0, distill_prob)))
        if distill_prob <= 0:
            return x_normal_t.new_zeros(())
        if distill_prob < 1.0 and torch.rand(1, device=x_0.device).item() >= distill_prob:
            return x_normal_t.new_zeros(())

        noisier_t = self._sample_single_path_distill_noisier_t(normal_t, args)

        with torch.no_grad():
            teacher_noise = self.noise_fn(x_0, noisier_t).float()
            x_noisier_t = self.sample_q(x_0, noisier_t, teacher_noise)
            estimate_noise_noisier = model(x_noisier_t, noisier_t)
            pred_x_0_noisier = self.predict_x_0_from_eps(
                x_noisier_t, noisier_t, estimate_noise_noisier
            ).clamp(-1, 1)
            pred_x_t_teacher = self.sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)
            teacher_residual = pred_x_t_teacher - x_normal_t

        student_residual = pred_x_t_proxy - x_normal_t

        if self._get_arg_bool(args, "single_path_distill_normalize", True):
            residual_norm = extract(
                self.sqrt_one_minus_alphas_cumprod,
                normal_t,
                x_normal_t.shape,
                x_0.device,
            )
            norm_eps = self._get_arg_float(args, "single_path_distill_eps", 1e-6)
            student_residual = student_residual / (residual_norm + norm_eps)
            teacher_residual = teacher_residual / (residual_norm + norm_eps)

        per_sample_loss = mean_flat((student_residual - teacher_residual).square())

        normal_mask = anomaly_label == 0
        if normal_mask.ndim == 0:
            normal_mask = normal_mask.unsqueeze(0)

        if normal_mask.any():
            return per_sample_loss[normal_mask].mean()
        return x_normal_t.new_zeros(())

   
    def norm_guided_one_step_denoising(self, model, x_0, anomaly_label,args):
        # two-scale t
        normal_t = torch.randint(0, args["less_t_range"], (x_0.shape[0],),device=x_0.device)
        normal_loss, x_normal_t, estimate_noise_normal = self.calc_loss(model, x_0, normal_t)

        if self._get_arg_bool(args, "diffusion_single_path", False):
            loss_scale = self._get_arg_float(args, "single_path_loss_scale", 2.0)
            condition_scale = self._get_arg_float(args, "single_path_condition_w_scale", 1.0)

            pred_x_0_normal, pred_x_t_proxy, proxy_state_t, _pred_x_0_proxy = self._build_single_path_proxy(
                model, x_normal_t, normal_t, estimate_noise_normal, args
            )

            residual_scale = self._get_arg_float(args, "single_path_residual_scale", 1.0)
            residual = (pred_x_t_proxy - x_normal_t) * residual_scale
            residual = self._apply_consistency_guidance_weight(
                model,
                residual,
                x_normal_t,
                normal_t,
                estimate_noise_normal,
                args,
            )
            residual = self._apply_irf_guidance_weight(
                residual,
                normal_t,
                args,
                anomaly_label=anomaly_label,
            )

            guidance_scale = self._build_guidance_scale(
                normal_t,
                args,
                x_normal_t,
                base_scale=self._get_arg_float(args, "condition_w", 1.0) * condition_scale,
            )

            estimate_noise_hat = estimate_noise_normal - extract(
                self.sqrt_one_minus_alphas_cumprod,
                normal_t,
                x_normal_t.shape,
                x_0.device,
            ) * guidance_scale * residual

            pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

            # Approximate the two-path training loss with scaled single-path noise loss.
            normal_noise_loss = (normal_loss["loss"] * loss_scale)[anomaly_label == 0].mean()
            distill_loss = self._calc_single_path_residual_distill_loss(
                model,
                x_0,
                x_normal_t,
                normal_t,
                estimate_noise_normal,
                pred_x_t_proxy,
                anomaly_label,
                args,
            )
            distill_weight = self._get_arg_float(args, "single_path_distill_weight", 0.0)
            loss = normal_noise_loss + distill_weight * distill_loss
            if torch.isnan(loss):
                loss.fill_(0.0)

            return loss, pred_x_0_norm_guided, normal_t, x_normal_t, proxy_state_t

        noisier_t = torch.randint(args["less_t_range"],self.num_timesteps,(x_0.shape[0],),device=x_0.device)
        noisier_loss, x_noiser_t, estimate_noise_noisier = self.calc_loss(model, x_0, noisier_t)
        
        pred_x_0_noisier = self.predict_x_0_from_eps(x_noiser_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)   

        # Only calculate the noise loss of normal samples according to formula 9.
        loss = (normal_loss["loss"]+noisier_loss["loss"])[anomaly_label==0].mean()
        # When the batch size is small, it may lead to an entire batch consisting solely of abnormal samples
        # If they are all abnormal samples, set loss to 0.
        if torch.isnan(loss):
            loss.fill_(0.0)

        guidance_scale = self._build_guidance_scale(
            normal_t,
            args,
            x_normal_t,
            base_scale=self._get_arg_float(args, "condition_w", 1.0),
        )
        estimate_noise_hat = estimate_noise_normal - extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_normal_t.shape, x_0.device) * guidance_scale * (pred_x_t_noisier-x_normal_t)
        pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

        return loss,pred_x_0_norm_guided,normal_t,x_normal_t,x_noiser_t


    def norm_guided_one_step_denoising_eval(self, model, x_0, normal_t,noisier_t,args):

        
        normal_loss, x_normal_t, estimate_noise_normal = self.calc_loss(model, x_0, normal_t)

        if self._get_arg_bool(args, "diffusion_single_path", False):
            loss_scale = self._get_arg_float(args, "single_path_loss_scale", 2.0)
            condition_scale = self._get_arg_float(args, "single_path_condition_w_scale", 1.0)

            loss = (normal_loss["loss"] * loss_scale).mean()
            pred_x_0_normal, pred_x_t_proxy, proxy_state_t, pred_x_0_proxy = self._build_single_path_proxy(
                model, x_normal_t, normal_t, estimate_noise_normal, args
            )

            residual_scale = self._get_arg_float(args, "single_path_residual_scale", 1.0)
            residual = (pred_x_t_proxy - x_normal_t) * residual_scale
            residual = self._apply_consistency_guidance_weight(
                model,
                residual,
                x_normal_t,
                normal_t,
                estimate_noise_normal,
                args,
            )
            residual = self._apply_irf_guidance_weight(
                residual,
                normal_t,
                args,
                anomaly_label=None,
            )

            guidance_scale = self._build_guidance_scale(
                normal_t,
                args,
                x_normal_t,
                base_scale=self._get_arg_float(args, "condition_w", 1.0) * condition_scale,
            )

            estimate_noise_hat = estimate_noise_normal - extract(
                self.sqrt_one_minus_alphas_cumprod,
                normal_t,
                x_0.shape,
                x_0.device,
            ) * guidance_scale * residual
            pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

            return (
                loss,
                pred_x_0_norm_guided,
                pred_x_0_normal,
                pred_x_0_proxy,
                x_normal_t,
                proxy_state_t,
                pred_x_t_proxy,
            )

        noisier_loss, x_noisier_t, estimate_noise_noisier = self.calc_loss(model, x_0, noisier_t)

        pred_x_0_noisier = self.predict_x_0_from_eps(x_noisier_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)    

        loss = (normal_loss["loss"]+noisier_loss["loss"]).mean()
        pred_x_0_normal = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_normal).clamp(-1, 1)
        guidance_scale = self._build_guidance_scale(
            normal_t,
            args,
            x_normal_t,
            base_scale=self._get_arg_float(args, "condition_w", 1.0),
        )
        estimate_noise_hat = estimate_noise_normal - extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_0.shape, x_0.device) * guidance_scale * (pred_x_t_noisier-x_normal_t)
        pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)
        
        return loss,pred_x_0_norm_guided,pred_x_0_normal,pred_x_0_noisier,x_normal_t,x_noisier_t,pred_x_t_noisier  
    

    def noise_t(self, model, x_0, t,args):
        loss, x_t, estimate_noise = self.calc_loss(model, x_0, t)
        loss = (loss["loss"] ).mean()
        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        return loss,pred_x_0,x_t

