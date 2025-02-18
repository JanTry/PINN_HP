import torch
import torch.nn.functional as F


def get_mesh_1D(x, x_ini, x_fin, reorder=True):
    local_x = F.pad(x, pad=(0, 1), mode="constant", value=x_ini)
    local_x = F.pad(local_x, pad=(0, 1), mode="constant", value=x_fin)

    idx = torch.argsort(local_x, dim=-1)
    a = torch.arange(len(idx) - 1)
    b = torch.arange(1, len(idx))
    conecs = torch.stack((idx[a], idx[b]), dim=0).transpose(1, 0)

    coords = local_x.reshape(-1, 1)

    if reorder:
        coords = coords[torch.cat((conecs[:, 0], conecs[-1, -1].reshape(-1)))]
        a = torch.arange(len(coords), device=x.device)
        conecs = torch.stack((a[0:-1], a[1:])).transpose(1, 0)

    return coords, conecs
