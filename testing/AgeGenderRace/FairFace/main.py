import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import models
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        xy = np.loadtxt(csv_file, delimiter=',', dtype=str, skiprows=1)
        self.paths = xy[:, 0]
        self.ages = xy[:, 1]
        self.genders = xy[:, 2]
        self.races = xy[:, 3]
        self.n_samples = xy.shape[0]
        self.transform = transform
        
        self.age_mapping = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8
        }
        
        self.gender_mapping = {
            'Male': 0,
            'Female': 1
        }
        
        self.race_mapping = {
            'White': 0,
            'Black': 1,
            'Latino_Hispanic': 2,
            'East Asian': 3,
            'Southeast Asian': 4,
            'Indian': 5,
            'Middle Eastern': 6
        }
        
    def __getitem__(self, index):
        path = self.paths[index]
        age = self.age_mapping[self.ages[index]]
        gender = self.gender_mapping[self.genders[index]]
        race = self.race_mapping[self.races[index]]
        
        image = Image.open(path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, age, gender, race
    
    def __len__(self):
        return self.n_samples


# Hyperparameters
learning_rate = 0.01
epochs = 50
batch_size = 64

# Transformations for the images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset and DataLoader
train_dataset = FairFaceDataset('fairface_label_train.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = FairFaceDataset('fairface_label_val.csv', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Train Dataset: {len(train_dataset)} samples')
print(f'Validation Dataset: {len(val_dataset)} samples')


#Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.fc_age = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )
        self.fc_gender = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.fc_race = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
        
    def forward(self, x):
        x = self.model(x)
        age = self.fc_age(x)
        gender = self.fc_gender(x)
        race = self.fc_race(x)
        return age, gender, race

model = ConvNet()
criteria = (nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

val_gender_accuracies = []
val_age_accuracies = []
val_race_accuracies = []
train_losses = []
val_losses = []

n_epochs = epochs
best_test_loss = float('inf')
start = time.time()

def train_batch(data, model, optimizer, criteria):
    model.train()
    ims, age, gender, race = data
    ims, age, gender, race = ims.cuda(), age.cuda(), gender.cuda(), race.cuda()
    optimizer.zero_grad()
    pred_age, pred_gender, pred_race = model(ims)
    age_criterion, gender_criterion, race_criterion = criteria
    age_loss = age_criterion(pred_age, age)
    gender_loss = gender_criterion(pred_gender, gender)
    race_loss = race_criterion(pred_race, race)
    total_loss = age_loss + gender_loss + race_loss
    total_loss.backward()
    optimizer.step()
    return total_loss

def validate_batch(data, model, criteria):
    model.eval()
    ims, age, gender, race = data
    ims, age, gender, race = ims.cuda(), age.cuda(), gender.cuda(), race.cuda()
    with torch.no_grad():
        pred_age, pred_gender, pred_race = model(ims)
    age_criterion, gender_criterion, race_criterion = criteria
    age_loss = age_criterion(pred_age, age)
    gender_loss = gender_criterion(pred_gender, gender)
    race_loss = race_criterion(pred_race, race)
    total_loss = age_loss + gender_loss + race_loss
    pred_age = torch.argmax(pred_age, dim=1)
    pred_gender = torch.argmax(pred_gender, dim=1)
    pred_race = torch.argmax(pred_race, dim=1)
    age_acc = (pred_age == age).float().mean().item()
    gender_acc = (pred_gender == gender).float().mean().item()
    race_acc = (pred_race == race).float().mean().item()
    return total_loss, age_acc, gender_acc, race_acc

model.cuda()

for epoch in range(n_epochs):
    epoch_train_loss, epoch_test_loss = 0, 0
    val_age_acc, val_gender_acc, val_race_acc = 0, 0, 0

    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        loss = train_batch(data, model, optimizer, criteria)
        epoch_train_loss += loss.item()
    
    for data in val_loader:
        loss, age_acc, gender_acc, race_acc = validate_batch(data, model, criteria)
        epoch_test_loss += loss.item()
        val_age_acc += age_acc
        val_gender_acc += gender_acc
        val_race_acc += race_acc
    
    epoch_train_loss /= len(train_loader)
    epoch_test_loss /= len(val_loader)
    val_age_acc /= len(val_loader)
    val_gender_acc /= len(val_loader)
    val_race_acc /= len(val_loader)
    best_test_loss = min(best_test_loss, epoch_test_loss)
    
     # Update learning rate
    scheduler.step(epoch_test_loss)
    
    elapsed = time.time() - start
    print(f'Epoch {epoch+1}/{n_epochs} ({elapsed:.2f}s elapsed) - '
          f'Train Loss: {epoch_train_loss:.3f} - Test Loss: {epoch_test_loss:.3f} - '
          f'Best Test Loss: {best_test_loss:.3f} - Age Accuracy: {val_age_acc:.2f} - '
          f'Gender Accuracy: {val_gender_acc:.2f} - Race Accuracy: {val_race_acc:.2f}'
          f'Current LR: {optimizer.param_groups[0]["lr"]}')

    val_age_accuracies.append(val_age_acc)
    val_gender_accuracies.append(val_gender_acc)
    val_race_accuracies.append(val_race_acc)
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_test_loss)
    
    
epoch_range = range(1, n_epochs+1)
    
plt.plot(epoch_range, val_age_accuracies, label='Age Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Age Accuracy')
plt.legend()
plt.savefig('ageaccuracies.png')
plt.clf()  # Bereinigt den aktuellen Plot

# Plot für Gender Accuracy
plt.plot(epoch_range, val_gender_accuracies, label='Gender Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Gender Accuracy')
plt.legend()
plt.savefig('genderaccuracies.png')
plt.clf()  # Bereinigt den aktuellen Plot

# Plot für Race Accuracy
plt.plot(epoch_range, val_race_accuracies, label='Race Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Race Accuracy')
plt.legend()
plt.savefig('raceaccuracies.png')
plt.clf() 

# Save the trained model
torch.save(model.state_dict(), 'fairface_model_greyscale.pth')
print('Model saved as fairface_model.pth')
