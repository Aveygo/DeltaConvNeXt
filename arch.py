import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class DualHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.left = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.right = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.scaling = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return (left + right) * (1 / (1 + torch.exp((left-right)**2) * self.scaling))

class DeltaConvNeXt(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.dwconv = DualHead(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.grn = GRN(2 * dim)
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x_in):
        x = self.dwconv(x_in)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return self.drop_path(x_in) + x


if __name__ == "__main__":
    m = DeltaConvNeXt(768).train().cuda()
    y = m(torch.randn((1, 768, 32, 32), device="cuda"))
    print(y.shape) # torch.Size([1, 768, 32, 32])

    from torchstat import stat
    stat(m.cpu(), (768, 32, 32))
    
    """
    Total params: 2,443,008
    -------------------------
    Total memory: 33.00MB
    Total MAdd: 154.14MMAdd
    Total Flops: 78.65MFlops
    Total MemR+W: 39.3MB
    """
