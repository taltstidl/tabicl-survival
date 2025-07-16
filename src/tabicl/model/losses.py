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


def mean_squared_error_and_rank(estimate: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """ Extended mean squared error and pairwise ranking loss.
    From RankDeepSurv: https://doi.org/10.1016/j.artmed.2019.06.001

    """
    loss1 = torch.mean(torch.square(estimate - time) * (event | (estimate <= time)).float())
    # TODO: implement pairwise ranking loss
    return loss1
