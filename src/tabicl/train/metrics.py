import torch


def concordance_index(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    risk_comp = risk.unsqueeze(-1) < risk.unsqueeze(-2)
    time_comp = time.unsqueeze(-1) > time.unsqueeze(-2)
    event = event.unsqueeze(-2)
    return torch.sum(risk_comp & time_comp & event) / torch.sum(time_comp & event)
