import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class VideoResNet(nn.Module):
    """
    Video classification model based on ResNet backbone.
    This model extracts features from video frames using ResNet,
    then aggregates them temporally for classification.
    """

    def __init__(
        self,
        num_classes=10,
        backbone="resnet50",
        pretrained=True,
        pool_type="avg",
        dropout_prob=0.5,
    ):
        """
        Initialize the model.

        Args:
            num_classes: Number of output classes
            backbone: ResNet variant to use ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use pretrained weights
            pool_type: Type of temporal pooling ('avg', 'max', 'attention')
            dropout_prob: Dropout probability before final classification layer
        """
        super(VideoResNet, self).__init__()

        # Load the specified ResNet backbone
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet34":
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            base_model = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet152":
            base_model = models.resnet152(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Temporal pooling
        self.pool_type = pool_type
        if pool_type == "attention":
            self.attention = nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(feature_dim // 8, 1, kernel_size=1),
                nn.Softmax(dim=-1),
            )

        # Final classifier
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, num_frames, channels, height, width]

        Returns:
            Class predictions tensor
        """
        batch_size, num_frames, c, h, w = x.shape

        # Reshape for frame-wise processing
        x = x.view(-1, c, h, w)

        # Extract features using ResNet backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Reshape back to [batch_size, num_frames, feature_dim]
        features = features.view(batch_size, num_frames, -1)

        # Temporal pooling
        if self.pool_type == "avg":
            features = torch.mean(features, dim=1)
        elif self.pool_type == "max":
            features = torch.max(features, dim=1)[0]
        elif self.pool_type == "attention":
            # Transpose for 1D convolution [batch, channels, time]
            attention_features = features.transpose(1, 2)
            attention_weights = self.attention(attention_features)

            # Apply attention weights
            attention_weights = attention_weights.view(batch_size, 1, num_frames)
            features = torch.bmm(attention_weights, features).squeeze(1)

        # Final classification
        features = self.dropout(features)
        output = self.classifier(features)

        return output


class VideoModel(nn.Module):
    def __init__(
        self, num_classes=10, model_name="r3d_18", pretrained=True, dropout_prob=0.5
    ):
        """
        Initialize the 3D video classification model.

        Args:
            num_classes: Number of output classes
            model_name: Name of the 3D model from torchvision.models.video
            pretrained: Whether to use pretrained weights
            dropout_prob: Dropout probability before final classification layer
        """
        super(VideoModel, self).__init__()

        # Load the specified 3D model from torchvision
        if model_name not in models.video.__dict__:
            raise ValueError(f"Unsupported model_name: {model_name}")
        base_model = models.video.__dict__[model_name](pretrained=pretrained)

        # Replace the classifier head
        feature_dim = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(dropout_prob), nn.Linear(feature_dim, num_classes)
        )

        self.model = base_model

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, channels, num_frames, height, width]

        Returns:
            Class predictions tensor
        """
        batch_size, num_frames, c, h, w = x.shape

        x = x.permute(
            0, 2, 1, 3, 4
        )  # Change to [batch_size, channels, num_frames, height, width]
        # Ensure the input is in the correct format for the model
        # if x.dim() != 5:
        #     raise ValueError(f"Input tensor must have 5 dimensions, got {x.dim()} instead.")
        # if x.shape[1] != 3:
        #     raise ValueError(f"Input tensor must have 3 channels, got {x.shape[1]} instead.")
        # if x.shape[2] != 16:
        #     raise ValueError(f"Input tensor must have 16 frames, got {x.shape[2]} instead.")
        # if x.shape[3] != 112 or x.shape[4] != 112:
        #     raise ValueError(f"Input tensor must have height and width of 112, got {x.shape[3]}x{x.shape[4]} instead.")
        # # Reshape the input tensor to match the model's expected input shape

        return self.model(x)


def get_video_classifier(num_classes, backbone="resnet50", **kwargs):
    """
    Factory function for creating video classification models.

    Args:
        num_classes: Number of classes to classify
        backbone: Model backbone to use ('resnet18', 'resnet34', 'resnet50',
                 'resnet101', 'resnet152', 'r3d_18', 'r2plus1d_18')
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Initialized video classification model
    """
    if backbone in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        return VideoResNet(num_classes=num_classes, backbone=backbone, **kwargs)
    elif backbone in ["r3d_18", "r2plus1d_18"]:
        return VideoModel(num_classes=num_classes, model_name=backbone, **kwargs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
