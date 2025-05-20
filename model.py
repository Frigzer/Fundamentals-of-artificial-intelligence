# model.py

import torch.nn as nn
import torch.nn.functional as F


class ASLClassifier(nn.Module):
    def __init__(self, num_classes=36):
        super(ASLClassifier, self).__init__()

        # Konwolucja
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Pierwszy etap - znajduje linie krawędzie

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,
                               padding=1)  # Drugi etap - znajduje kształy np. palce, kontury dłoni
        self.pool = nn.MaxPool2d(2, 2)  # Zmniejsza rozdzielczość, rozmiar obrazu na pół

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,
                               padding=1)  # Trzeci etap - znajduje całe gesty np. litera A, B, C

        self.dropout = nn.Dropout(0.3)  # Zmniejsza liczbę neuronów o 30%
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # zakładając wejście 224x224, po spłaszczeniu łączy w 512 neuronów
        self.fc2 = nn.Linear(512, num_classes)  # Po jednym na klasę

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112x112
        x = self.pool(F.relu(self.conv2(x)))  # 56x56
        x = self.pool(F.relu(self.conv3(x)))  # 28x28
        x = x.view(x.size(0), -1)  # przekształcamy obraz w wektor
        '''
        To jest flatten – spłaszcza dane przed podaniem do warstwy liniowej (fc1).
        Potrzebne, bo nn.Linear() nie umie działać na obrazach 3D (czyli [batch_size, channels, H, W]).
        '''
        x = self.dropout(F.relu(self.fc1(x)))  # wykrywanie cech, "redukuje szum", kompresuje dane
        x = self.fc2(x)  # ostatnia warstwa, czyli wyjście
        return x
