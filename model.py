"""A plain residual U-Net: 3 bright-field planes in, 5 Cell Painting channels out."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.silu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018): the decoder signal decides which skip pixels pass."""

    def __init__(self, ch):
        super().__init__()
        inter = max(ch // 2, 8)
        self.wg, self.wx = nn.Conv2d(ch, inter, 1), nn.Conv2d(ch, inter, 1)
        self.psi = nn.Sequential(nn.SiLU(), nn.Conv2d(inter, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(self.wg(g) + self.wx(x))


def up_layer(cin, cout, mode):
    """Transposed conv is a bare layer so existing checkpoints (keys up.N.weight) keep loading."""
    if mode == "bilinear":
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(cin, cout, 1))
    return nn.ConvTranspose2d(cin, cout, 2, stride=2)


class UNet(nn.Module):
    """Residual U-Net. Defaults reproduce the original model (width 32, depth 4, transposed up-sampling)."""

    def __init__(self, cin=3, cout=5, base=32, depth=4, attention=False, upsampling="transpose"):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.stem = Block(cin, chs[0])
        self.down = nn.ModuleList(Block(chs[i], chs[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(up_layer(chs[i + 1], chs[i], upsampling) for i in reversed(range(depth)))
        self.gates = nn.ModuleList(AttentionGate(chs[i]) for i in reversed(range(depth))) if attention else None
        self.dec = nn.ModuleList(Block(chs[i] * 2, chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], cout, 1)
        self.depth = depth

    def forward(self, x):
        # Pad to a multiple of 2**depth so whole 540 px fields can be predicted in one pass.
        H, W = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-H) % m, (-W) % m
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            h = up(h)
            skip = skips.pop()
            if self.gates is not None:
                skip = self.gates[i](h, skip)
            h = dec(torch.cat([h, skip], 1))
        return self.head(h)[..., :H, :W]


class PaddedSMP(nn.Module):
    """segmentation_models_pytorch U-Net with a pretrained encoder; pads input to a multiple of 32."""

    def __init__(self, cin, cout, encoder="resnet34", attention=False, pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=cin,
                            classes=cout, decoder_attention_type="scse" if attention else None)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32
        return self.net(F.pad(x, (0, pw, 0, ph), mode="reflect"))[..., :H, :W]


def build_model(cfg, cin, cout, pretrained=True):
    """From a run config (train_log.json args or a search path). Old configs only have 'base'."""
    if cfg.get("model", "unet") == "unet_resnet34_pretrained":
        return PaddedSMP(cin, cout, "resnet34", attention=cfg.get("attention", False), pretrained=pretrained)
    return UNet(cin, cout, base=cfg.get("width", cfg.get("base", 32)), depth=cfg.get("depth", 4),
                attention=cfg.get("attention", False), upsampling=cfg.get("upsampling", "transpose"))


class PatchDiscriminator(nn.Module):
    """pix2pix-style PatchGAN (~70 px receptive field), conditioned on the bright-field input.
    InstanceNorm as in pix2pix (spectral norm crashes on CUDA in this torch build)."""

    def __init__(self, cin, ch=64):
        super().__init__()
        layers, c = [], cin
        for i, (o, s) in enumerate([(ch, 2), (ch * 2, 2), (ch * 4, 2), (ch * 8, 1)]):
            layers.append(nn.Conv2d(c, o, 4, s, 1))
            if i > 0:
                layers.append(nn.InstanceNorm2d(o, affine=True))
            layers.append(nn.LeakyReLU(0.2))
            c = o
        layers.append(nn.Conv2d(c, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, bf, stains):
        return self.net(torch.cat([bf, stains], 1))
