import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from configs.config import IMAGE_FEAT_WEIGHT
from src.heatmap_features import STRUCT_FEAT_DIM

# ======================================================================
# 5. PriorFusionB
# ======================================================================
def spatial_softmax_2d(hm, temperature=1.0, eps=1e-6):
    B, C, H, W = hm.shape
    x = hm.reshape(B, C, -1) / temperature
    x = x - x.max(dim=-1, keepdim=True).values
    w = torch.softmax(x, dim=-1).reshape(B, C, H, W)
    w = w.clamp_min(eps)
    w = w / (w.sum(dim=(2, 3), keepdim=True) + eps)
    return w


class PriorFusionB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.temperature = 1.0

        self.mlp = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, channels * 2)
        )

        self.hm_to_c = nn.Sequential(
            nn.Conv2d(1, channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, hm):
        B, C, H, W = x.shape

        if hm.shape[-2:] != (H, W):
            hm = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)

        w = spatial_softmax_2d(hm, temperature=self.temperature)

        ctx = (x.unsqueeze(1) * w.unsqueeze(2)).sum((3, 4)).mean(1)

        g, b = self.mlp(ctx).chunk(2, 1)

        y = x * (1 + g.unsqueeze(-1).unsqueeze(-1)) + b.unsqueeze(-1).unsqueeze(-1)

        y = y + 0.5 * self.hm_to_c(hm)

        return self.norm(y)

# ======================================================================
# 6. Model
# ======================================================================
def make_backbone():
    r = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return nn.Sequential(
        r.conv1, r.bn1, r.relu, r.maxpool,
        r.layer1, r.layer2, r.layer3, r.layer4
    )


# ======================================================================
# 6A. Image-only model
# ======================================================================
class TwoViewNetB_ImageOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = make_backbone()
        self.prior = PriorFusionB(512)
        self.attn = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x1, x2, h1, h2):
        f1 = self.prior(self.backbone(x1), h1)
        f2 = self.prior(self.backbone(x2), h2)
        f1 = F.adaptive_avg_pool2d(f1, 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)
        a = self.attn(torch.cat([f1, f2], 1))
        f = a * f1 + (1 - a) * f2
        return self.cls(f)


# ======================================================================
# 6B. Fusion model: 512 image features + 512 structural features -> 1024 classifier
# ======================================================================
class TwoViewNetB_Fusion(nn.Module):
    def __init__(self, image_weight=IMAGE_FEAT_WEIGHT):
        super().__init__()

        self.image_weight = float(image_weight)
        self.struct_weight = 1.0 - self.image_weight

        self.backbone = make_backbone()
        self.prior = PriorFusionB(512)

        self.attn = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.struct_mlp = nn.Sequential(
            nn.Linear(STRUCT_FEAT_DIM * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

        self.cls = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, 2)
        )

    def forward(self, x1, x2, h1, h2, struct1, struct2):
        # --------------------------------------------------------------
        # 1) Image branch: ResNet18 + PriorFusionB
        # --------------------------------------------------------------
        f1 = self.prior(self.backbone(x1), h1)
        f2 = self.prior(self.backbone(x2), h2)

        f1 = F.adaptive_avg_pool2d(f1, 1).flatten(1)  # [B,512]
        f2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)  # [B,512]

        a = self.attn(torch.cat([f1, f2], 1))
        img_feat = a * f1 + (1 - a) * f2  # [B,512]

        # --------------------------------------------------------------
        # 2) Structural branch: heatmap point count, coordinates, and spacing -> 512 dimensions
        # --------------------------------------------------------------
        struct = torch.cat([struct1, struct2], dim=1)
        struct_feat = self.struct_mlp(struct)  # [B,512]

        # --------------------------------------------------------------
        # 3) Weighted fusion
        # --------------------------------------------------------------
        img_feat = self.image_weight * img_feat
        struct_feat = self.struct_weight * struct_feat

        fused_feat = torch.cat([img_feat, struct_feat], dim=1)  # [B,1024]

        return self.cls(fused_feat)

# ======================================================================
# 6.1 Two-stage fine-tuning helper functions
# ======================================================================
def freeze_backbone_except_layer4_imageonly(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for p in model.backbone[-1].parameters():
        p.requires_grad = True

    for p in model.prior.parameters():
        p.requires_grad = True
    for p in model.attn.parameters():
        p.requires_grad = True
    for p in model.cls.parameters():
        p.requires_grad = True


def freeze_backbone_except_layer4_fusion(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for p in model.backbone[-1].parameters():
        p.requires_grad = True

    for p in model.prior.parameters():
        p.requires_grad = True
    for p in model.attn.parameters():
        p.requires_grad = True
    for p in model.struct_mlp.parameters():
        p.requires_grad = True
    for p in model.cls.parameters():
        p.requires_grad = True


def unfreeze_all_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = True

# ======================================================================
# 6.2 BN stability helper function
# ======================================================================
def set_backbone_bn_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()