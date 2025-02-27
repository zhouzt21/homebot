import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


class ResnetBackbone(nn.Module):
    def __init__(self, emb_dim, name: str = "resnet18"):
        super().__init__()
        self.model: torchvision.models.ResNet = getattr(torchvision.models, name)(
            pretrained=True
        )
        out_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.projector = nn.Linear(out_dim, emb_dim)

    def forward(self, image):
        return self.projector(self.model(self.transform(image / 255.0)))

    def forward_feat_before_norm(self, image):
        return self.model(self.transform(image / 255.0))

    def forward_feat_learnable(self, feat):
        return self.projector(feat)
