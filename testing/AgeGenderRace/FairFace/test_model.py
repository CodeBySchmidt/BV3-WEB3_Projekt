import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from main import ConvNet 

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your own image
image_path = '----.jpg'  # Replace with your image path
image = Image.open(image_path)

# Apply the transformations
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension


model = ConvNet()
model.load_state_dict(torch.load('fairface_model_grayscale.pth'))
model.cuda()
model.eval()

with torch.no_grad():
    image = image.cuda()
    pred_age, pred_gender, pred_race = model(image)


pred_age = torch.argmax(pred_age, dim=1).item()
pred_gender = torch.argmax(pred_gender, dim=1).item()
pred_race = torch.argmax(pred_race, dim=1).item()


age_mapping = {
    0: '0-2',
    1: '3-9',
    2: '10-19',
    3: '20-29',
    4: '30-39',
    5: '40-49',
    6: '50-59',
    7: '60-69',
    8: 'more than 70'
}

gender_mapping = {
    0: 'Male',
    1: 'Female'
}

race_mapping = {
    0: 'White',
    1: 'Black',
    2: 'Latino_Hispanic',
    3: 'East Asian',
    4: 'Southeast Asian',
    5: 'Indian',
    6: 'Middle Eastern'
}

# Convert predictions to labels
pred_age_label = age_mapping[pred_age]
pred_gender_label = gender_mapping[pred_gender]
pred_race_label = race_mapping[pred_race]

print(f'Predicted Age: {pred_age_label}')
print(f'Predicted Gender: {pred_gender_label}')
print(f'Predicted Race: {pred_race_label}')

# Optionally, display the image
plt.imshow(Image.open(image_path))
plt.title(f'Predicted Age: {pred_age_label}, Gender: {pred_gender_label}, Race: {pred_race_label}')
plt.show()
