from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.optim import SGD
from timeit import default_timer as timer
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Umschalten zwischen Colab oder lokaler Installation
USING_COLAB = False

# if USING_COLAB:
#     from google.colab import drive
#     drive.mount('/content/drive')

# --- Datensatz vorbereiten ---
train_set = datasets.MNIST('data/', download=True, train=True)
train_images = train_set.data
train_targets = train_set.targets

test_set = datasets.MNIST('data/', download=True, train=False)
test_images = test_set.data
test_targets = test_set.targets


class MNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / 255
        x = x.view(-1, 28 * 28)
        self.x, self.y = x, y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)


def get_data():
    train = MNISTDataset(train_images, train_targets)
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test = MNISTDataset(test_images, test_targets)
    test_dl = DataLoader(test, batch_size=len(test_images), shuffle=True)
    return train_dl, test_dl


def get_model():
    model = nn.Sequential(
        nn.Linear(28 * 28, 30),
        nn.Tanh(),
        nn.Linear(30, 20),
        nn.Tanh(),
        nn.Linear(20, 10)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer


def init_weights(m):
    if nn.Linear == type(m):
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)


# --- Training ---
def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


def accuracy(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


def loss(x, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        loss = loss_fn(prediction, y)
    return loss.item()


def train_model():
    # Daten laden
    train_dl, test_dl = get_data()
    model, loss_fn, optimizer = get_model()  # Modell, Verlustfunktion und Optimizer abrufen

    print('Starting training...')
    model.apply(init_weights)  # Initial weights

    epochs = 200
    arrPlotX = []
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(epochs):
        timeBeginEpoch = timer()
        train_epoch_losses, train_epoch_accuracies = [], []

        for ix, batch in enumerate(iter(train_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_loss = np.array(train_epoch_losses).mean()
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for ix, batch in enumerate(iter(test_dl)):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            validation_loss = loss(x, y, model, loss_fn)

        val_epoch_accuracy = np.mean(val_is_correct)
        arrPlotX.append(epoch)
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        test_losses.append(validation_loss)
        test_accuracies.append(val_epoch_accuracy)
        timeEndEpoch = timer()
        print(
            f"epoch: {epoch}  train_acc: {100 * train_epoch_accuracy:.2f}%  test_acc: {100 * val_epoch_accuracy:.2f}%  took {timeEndEpoch - timeBeginEpoch:.1f}s")

        # Modell speichern
    if USING_COLAB:
        torch.save(model.state_dict(), '/content/drive/My Drive/ColabNotebooks/results/nnMnist_exp01.pt')
    else:
        torch.save(model.state_dict(), 'nnMnist_exp01.pth')

    # --- Plot für die Genauigkeiten ---
    plt.figure()  # Erstelle eine neue Figur für die Genauigkeiten
    plt.plot(arrPlotX, train_accuracies, label='Train Accuracy')
    plt.plot(arrPlotX, test_accuracies, label='Test Accuracy')
    plt.title('Train and Test Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if USING_COLAB:
        plt.savefig('/content/drive/My Drive/ColabNotebooks/results/accuracies_exp0.png')
    else:
        plt.savefig('accuracies_exp0.png')

    # --- Plot für die Verluste ---
    plt.figure()  # Erstelle eine neue Figur für die Verluste
    plt.plot(arrPlotX, train_losses, label='Train Loss')
    plt.plot(arrPlotX, test_losses, label='Test Loss')
    plt.title('Train and Test Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if USING_COLAB:
        plt.savefig('/content/drive/My Drive/ColabNotebooks/results/losses_exp0.png')
    else:
        plt.savefig('losses_exp0.png')


def predict_single_image(model, image_path):
    # Lade das Bild und konvertiere es in Graustufen
    img = Image.open(image_path).convert('L')  # 'L' für Graustufen (1 Kanal)

    # Größe des Bildes auf 28x28 anpassen, falls es nicht schon so ist
    img = img.resize((28, 28))

    # Zeige das Bild zur Kontrolle
    plt.imshow(img, cmap='gray')
    plt.title("Eingabebild")
    plt.show()

    # Konvertiere das Bild in einen Tensor (Normalisierung auf [0, 1])
    img_array = np.array(img)  # Konvertiere Bild in NumPy Array
    img_tensor = torch.Tensor(img_array).unsqueeze(0).float() / 255.0  # [1, 28, 28]

    # Bild flach machen und in eine Batch von 1 packen
    img_tensor = img_tensor.view(1, 28 * 28).to(device)  # [1, 784] für das Netz

    # Modell auf Evaluationsmodus setzen
    model.eval()

    # Vorhersage durchführen
    with torch.no_grad():
        prediction = model(img_tensor.to(device))  # Ausgabe der Logits
        predicted_label = prediction.argmax(dim=-1).item()  # Nimm die Klasse mit dem höchsten Wert
        # print(f"Logits: {prediction}")
        print(f"Predicted Label: {predicted_label}")

    return predicted_label


# --- Laden des Modells und Vorhersage auf einem Bild ---
def load_model_and_predict(image_path):
    # Modell initialisieren
    model, _, _ = get_model()  # Initialisiere Modell ohne den Optimierer

    # Trainierte Gewichte laden
    model.load_state_dict(torch.load('nnMnist_exp01.pth', map_location=device))

    # check_model_weights(model)

    # Bild vorhersagen
    return predict_single_image(model, image_path)


# Überprüfe die Modell-Gewichte direkt nach dem Laden
def check_model_weights(model, path='nnMnist_exp01.pth'):
    model.load_state_dict(torch.load(path, map_location=device))
    print("Model weights loaded successfully.")
    for name, param in model.named_parameters():
        print(name, param.data.mean())


# --- Main Funktion ---
if __name__ == "__main__":
    mode = input("Möchtest du trainieren oder ein Bild vorhersagen? (train/predict): ").strip().lower()

    if mode == "train":
        train_model()
    elif mode == "predict":
        image_path = input("Gib den Pfad zu deinem Bild an: ")
        load_model_and_predict(image_path)
    else:
        print("Ungültige Eingabe. Bitte 'train' oder 'predict' eingeben.")
