import torch


def cox_neg_log_likelihood(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """ Cox negative log partial likelihood.
    From DeepSurv: https://doi.org/10.1186/s12874-018-0482-1

    """
    # Sort risk and events by time
    sort_idx = torch.argsort(time, descending=True)
    risk = torch.gather(risk, dim=-1, index=sort_idx)
    event = torch.gather(event, dim=-1, index=sort_idx)
    # Due to the sorting before, log_risk[i] = log(sum(e^risk[j=0:i]) with time[j] >= time[i]
    log_risk = torch.logcumsumexp(risk, dim=-1)
    likelihood = (risk - log_risk) * event.float()
    return - likelihood.sum(dim=-1) / event.sum(dim=-1)


def weibull_neg_log_likelihood(params: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    pass


def mse_with_pairwise_rank(estimate: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """ Extended mean squared error and pairwise ranking loss.
    From RankDeepSurv: https://doi.org/10.1016/j.artmed.2019.06.001

    """
    # First part of the loss function is a simple mean squared error
    error = time - estimate
    loss1 = torch.mean(torch.square(error) * (event | (estimate <= time)).float(), dim=-1)

    # Here, we add extra dimensions to enable pairwise comparisons of (i,j)
    event_i, event_j = event.unsqueeze(-2), event.unsqueeze(-1)
    time_i, time_j = time.unsqueeze(-2), time.unsqueeze(-1)
    # The compatibility matrix specifies which pairs (i,j) can be compared accounting for censoring
    comp = event_i & (event_j | (time_i <= time_j))  # matrix C in report

    # Second part of the loss function encourages correct ranking among compatible pairs based on relative distance
    # To save computations, we can reuse the errors needed for loss1 as
    # (time_j - time_i) - (estimate_j - estimate_i) = (time_j - estimate_j) - (time_i - estimate_i)
    error_i, error_j = error.unsqueeze(-2), error.unsqueeze(-1)
    diff = torch.clamp(error_j - error_i, min=0)  # matrix D in report, fused with condition
    loss2 = torch.sum(diff * comp, dim=(-2, -1)) / estimate.shape[-1]

    return loss1 + loss2
