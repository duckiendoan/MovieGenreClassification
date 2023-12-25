import torch.nn as nn
import torch

class JointModel(nn.Module):
    def __init__(self, image_model, text_model, num_classes=18):
        super(JointModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.f_image = nn.Linear(1000, 256)
        self.f_text = nn.Linear(text_model.config.hidden_size, 256)
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
            # nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, image):
        img_feature = self.image_model(image)
        text_feature = self.text_model(input_ids, attention_mask).last_hidden_state
        x1 = self.f_image(img_feature)
        x2 = self.f_text(text_feature[:, 0, :])
        x = torch.cat([x1, x2], dim=1)
        out = self.fc(x)
        return out
    
class JointModelv2(nn.Module):
    def __init__(self, image_model, text_model, num_classes=18):
        super(JointModelv2, self).__init__()
        self.image_model = image_model
        self.image_model.fc = nn.Identity()
        self.text_model = text_model
        self.f_image = nn.Linear(2048, 768)
        # self.f_text = nn.Linear(text_model.config.hidden_size, 256)
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(768 * 2, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            # nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, image):
        img_feature = self.image_model(image)
        text_feature = self.text_model(input_ids, attention_mask).last_hidden_state
        x1 = self.f_image(img_feature)
        # x2 = self.f_text(text_feature[:, 0, :])
        x2 = text_feature[:, 0, :]
        x = torch.cat([x1, x2], dim=1)
        out = self.fc(x)
        return out

class ImageOnlyModel(nn.Module):
    def __init__(self, image_model, text_model, num_classes=18):
        super(ImageOnlyModel, self).__init__()
        self.image_model = image_model
        self.image_model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
            # nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, image):
        x = self.image_model(image)
        out = self.fc(x)
        return out