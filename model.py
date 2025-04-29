import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program


# --- 1. Define your custom Dataset ---
class WeatherDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Load the CSV
        self.data = pd.read_csv(csv_file)

        # Remove rows with NaN in target columns
        self.data = self.data.dropna(subset=['TAVG', 'PRCP', 'SNOW'])
        # print(self.data)
        self.data = self.data[self.data['DATE'].astype(str).str.startswith(('2017', '2018', '2019',
                                                                            '2020', '2021', '2022',
                                                                            '2023', '2024', '2025'))]

        # Filter based on whether the corresponding image actually exists
        valid_rows = []

        for idx, row in self.data.iterrows():
            date = str(row['DATE'])
            year = date[:4]
            day_of_year = str(pd.to_datetime(date).dayofyear).zfill(3)

            img_path = os.path.join(
                self.img_dir,
                year,
                day_of_year,
                "12",
                f"truecolor{year}{day_of_year}.png"
            )
            if os.path.exists(img_path):
                valid_rows.append(row)

        # Only keep rows with existing images
        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        date = str(row['DATE'])
        year = date[:4]
        day_of_year = str(pd.to_datetime(date).dayofyear).zfill(3)
        img_path = os.path.join(
            self.img_dir,
            year,
            day_of_year,
            "12",
            f"truecolor{year}{day_of_year}.png"
        )

        # Load the image (guaranteed to exist)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get labels
        labels = torch.tensor([
            row['TAVG'],
            row['PRCP'],
            row['SNOW']
        ], dtype=torch.float32)

        return image, labels


class SequentialWeatherDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None, sequence_length=7, years=('2017', '2018', '2019',
                                                                                    '2020', '2021', '2022',
                                                                                    '2023')):
        self.img_dir = img_dir
        self.transform = transform
        self.sequence_length = sequence_length

        self.data = pd.read_csv(csv_file)

        self.data = self.data.dropna(subset=['TAVG', 'PRCP', 'SNOW'])

        self.data = self.data[self.data['DATE'].astype(str).str.startswith(years)]

        valid_rows = []

        for idx, row in self.data.iterrows():
            date = str(row['DATE'])
            year = date[:4]
            day_of_year = str(pd.to_datetime(date).dayofyear).zfill(3)

            img_path = os.path.join(
                self.img_dir,
                year,
                day_of_year,
                "12",
                f"truecolor{year}{day_of_year}.png"
            )
            if os.path.exists(img_path):
                valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length
        sequence_data = self.data.iloc[start_idx:end_idx]

        image_sequence = []
        labels_sequence = []

        for i, row in sequence_data.iterrows():
            date = str(row['DATE'])
            year = date[:4]
            day_of_year = str(pd.to_datetime(date).dayofyear).zfill(3)

            img_path = os.path.join(
                self.img_dir,
                year,
                day_of_year,
                "12",
                f"truecolor{year}{day_of_year}.png"
            )

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image: {img_path}") from e

            if self.transform:
                image = self.transform(image)

            image_sequence.append(image)

            labels_sequence.append(torch.tensor([
                row['TAVG'],
                row['PRCP'],
                row['SNOW']
            ], dtype=torch.float32))

        image_sequence = torch.stack(image_sequence)

        labels_sequence = torch.stack(labels_sequence)

        return image_sequence, labels_sequence[-1]

    # --- 4. Define a simple CNN Model ---


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Predicts TAVG, PRCP, SNOW
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(input_size=8192, hidden_size=128, num_layers=2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Predicts TAVG, PRCP, SNOW
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.features(x)
        x = x.view(batch_size, seq_len, -1)

        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


def train_run(model, num_epochs, dataset, sequence_length=7):
    writer = SummaryWriter()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    # --- 5. Instantiate Model, Loss, Optimizer ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch)
            # writer.add_scalar('Accuracy/test', np.random.random(), epoch)
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        writer.add_scalar("Total Loss/train", epoch_loss, epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if epoch % 10 == 0:
            eval_run(model, epoch, writer, sequence_length=sequence_length)
    writer.flush()
    writer.close()
    return model


def eval_run(model, epoch, writer, sequence_length=7):
    dataset = SequentialWeatherDataset(
        img_dir='Dataset/post_crop/ABI-L1b-RadM/',
        csv_file='Dataset/3997445.csv',
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]),
        years=('2024', '2025'),
        sequence_length=sequence_length
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss(reduction='none')

    model = model.to(device)
    model.eval()

    total_diff = torch.zeros(3, device=device)
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            diffs = criterion(outputs, labels)
            total_diff += diffs.sum(dim=0)
            total_samples += labels.size(0)

    avg_diff = total_diff / total_samples

    print(f"Average Difference per Variable: {avg_diff.cpu().numpy()}")
    # writer.add_scalar("Total Loss/train", , epoch)


if __name__ == '__main__':
    # --- 2. Setup transforms ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # --- 3. Create Dataset and DataLoader ---
    dataset = SequentialWeatherDataset(
        img_dir='Dataset/post_crop/ABI-L1b-RadM/',
        csv_file='Dataset/3997445.csv',
        transform=transform
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = SimpleCNN().to(device)
    model = LSTM().to(device)

    model = train_run(model, 11, dataset)
