# 🧠 American Sign Language Classifier

Projekt klasyfikatora obrazów znaków języka migowego ASL (American Sign Language) przy użyciu PyTorcha i CNN.  
Model trenuje na zbiorze obrazów przedstawiających znaki 0–9 oraz A–Z (łącznie 36 klas).

---

## 📁 Struktura katalogów

```
📦Fundamentals-of-artificial-intelligence/
 ┣ 📂data/
 ┃ ┣ 📂test/
 ┃ ┣ 📂train/
 ┃ ┣ 📂val/
 ┣ 📂serialized/
 ┃ ┣ 📂test/
 ┃ ┣ 📂train/
 ┃ ┣ 📂val/
 ┣ 📜prepare_data.py
 ┣ 📜chck_data.py
 ┣ 📜serialize_data.py
 ┣ 📜requirements.txt
 ┣ 📜.gitignore
 ┗ 📜README.md
```

---

## ⚙️ Wymagania

- Python 3.8+
- pip
- Git (lub GitHub Desktop)
- (opcjonalnie) VSCode z rozszerzeniem Python lub PyCharm

---

## 🚀 Uruchomienie projektu krok po kroku

### 1. Sklonuj repozytorium
```bash
git clone https://github.com/Frigzer/Fundamentals-of-artificial-intelligence.git
cd Fundamentals-of-artificial-intelligence
```

### 2. Utwórz środowisko wirtualne
```bash
python -m venv venv
```

### 3. 💡 [WAŻNE – tylko przy pierwszym użyciu PowerShella!]

Jeśli uruchomienie `venv\Scripts\activate` daje błąd "running scripts is disabled", wykonaj poniższe kroki w PowerShellu **uruchomionym jako Administrator**:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Potwierdź `[Y]`, zamknij i otwórz PowerShell ponownie.

### 4. Aktywuj środowisko
```bash
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 5. Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### 6. Przygotuj dane treningowe - WAŻNA KOLEJNOŚĆ!!!
```bash
python prepare_data.py
python check_data.py
python serialize_data.py
```

---

## 🖼️ Dane wejściowe

Obrazy muszą być posortowane w katalogach `train/<klasa>` i `test/<klasa>`, np.:

```
data/train/A/image1.jpg
data/train/B/image2.jpg
data/test/3/image5.jpg
```

---

## 📊 Rezultat

Program trenuje sieć konwolucyjną (CNN) i wypisuje:
- Straty treningowe
- Dokładność na zbiorze testowym
- Przykładowe predykcje (obrazy + etykiety)

---

## 🛡️ Info

Nie wrzucaj folderu `.venv/` do repo – jest ignorowany w `.gitignore`.  
Każdy powinien tworzyć własne środowisko lokalnie.
