import torch.nn as nn
import torch

class JointModel(nn.Module):
    def __init__(self, image_model, text_model, num_classes=18):
        super(JointModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.fc = nn.Sequential(
            nn.Linear(1000 + text_model.config.hidden_size, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, image):
        img_feature = self.image_model(image)
        text_feature = self.text_model(input_ids, attention_mask).last_hidden_state
        x = torch.cat([img_feature, text_feature[:, 0, :]], dim=1)
        out = self.fc(x)
        return out