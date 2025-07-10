# 🧠 American Sign Language Classifier

Projekt klasyfikatora obrazów znaków języka migowego ASL (American Sign Language) przy użyciu PyTorcha.  
Model bedzie się trenował na zbiorze obrazów przedstawiających znaki 0–9 oraz A–Z (łącznie 36 klas).

---

## 📁 Struktura katalogów

```
📦Fundamentals-of-artificial-intelligence/
 ┣ 📂data/
 ┃ ┣ 📂test/
 ┃ ┣ 📂train/
 ┃ ┣ 📂val/
 ┣ 📂photos/
 ┣ 📂serialized/
 ┃ ┣ 📂test/
 ┃ ┣ 📂train/
 ┃ ┣ 📂val/
 ┣ 📜prepare_data.py
 ┣ 📜check_data.py
 ┣ 📜serialize_data.py
 ┣ 📜run_model.py
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

## 🛡️ Info

Nie wrzucaj folderu `.venv/` oraz utworzonych folderów `data/` i `serialized/` do repo – są ignorowane w `.gitignore`.  
Każdy powinien tworzyć własne środowisko lokalnie.
