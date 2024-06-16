'''
ResNeXt for classifying sign language letters [a-z]
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.utils.data import DataLoader, Subset, ConcatDataset
# -- PyTorch Lightning
import lightning as L

# Set float32 matmul precision for Tensor Cores
torch.set_float32_matmul_precision('high')


class SignResNeXt(L.LightningModule):
    def __init__(self):

        super().__init__()

        # -- Network Parameters
        self.objective = nn.CrossEntropyLoss()
        self.batch_size = 32

        # -- Metrics
        a = 'macro'
        c = 26
        t = 'multiclass'
        
        self.accuracy = Accuracy(task=t, num_classes=c)
        self.f1 = F1Score(task=t, average=a, num_classes=c)
        self.precision = Precision(task=t, average=a, num_classes=c)
        self.recall = Recall(task=t, average=a, num_classes=c)

        # -- Network Architecture
        self.arch = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.arch.fc = nn.Linear(self.arch.fc.in_features, c)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):

        samples, labels = batch
        
        preds = self.arch(samples)

        loss = self.objective(preds, labels)

        self.log("train_error", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        
        samples, labels = batch

        # -- Predict
        preds = self.arch(samples)

        # -- Loss
        loss = self.objective(preds, labels)

        self.log("test_error", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        # Calculate: Confusion Matrix Analytics

        preds = torch.argmax(preds, dim=1)

        measures = {"accuracy": self.accuracy, "f1": self.f1,
                    "recall": self.recall, "precision": self.precision}

        for current_key in measures.keys():
            score = measures[current_key](preds, labels)
            self.log(current_key, score, batch_size=self.batch_size,
                     on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            
    def forward(self, samples):
        return self.arch(samples)
    
def load_classifier(path):

    model = SignResNeXt.load_from_checkpoint(path)

    model.eval()

    return model

def get_subset_indices(dataset, fraction):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    subset_size = int(fraction * dataset_size)
    return indices[:subset_size]

if __name__ == "__main__":

    '''
    Load data using torchvision.datasets.ImageFolder
    '''
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5)
    ])
    
    # -- Train dataset
    train_dataset1 = datasets.ImageFolder('../../SignChat_Data/ASL Alphabet/asl_alphabet_train/asl_alphabet_train', transform=transform)
    train_dataset1 = Subset(train_dataset1, get_subset_indices(train_dataset1, 1.0))
    train_dataset2 = datasets.ImageFolder('../../SignChat_Data/American Sign Language Letters.v1-v1.yolov8/train', transform=transform)
    train_dataset = ConcatDataset([train_dataset1, train_dataset2])
    class_to_idx = train_dataset2.class_to_idx
    # -- Test dataset
    test_dataset1 = datasets.ImageFolder('../../SignChat_Data/ASL Alphabet/asl_alphabet_test/asl_alphabet_test', transform=transform)
    test_dataset1 = Subset(train_dataset1, get_subset_indices(test_dataset1, 1.0))
    test_dataset2 = datasets.ImageFolder('../../SignChat_Data/American Sign Language Letters.v1-v1.yolov8/test', transform=transform)
    test_dataset = ConcatDataset([test_dataset1, test_dataset2])
    # -- Validation dataset
    val_dataset = datasets.ImageFolder('../../SignChat_Data/American Sign Language Letters.v1-v1.yolov8/valid', transform=transform)

    # -- Create subsets
    train_subset = Subset(train_dataset, get_subset_indices(train_dataset, 0.1))
    test_subset = Subset(test_dataset, get_subset_indices(test_dataset, 0.1))

    # -- Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # -- Initialize Trainer
    trainer = L.Trainer(max_epochs=5)

    # -- Create new model
    model = SignResNeXt()

    '''
    Train and test model
    '''
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)



