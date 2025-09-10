# Hussein-Jaafar-Thesis-2025 – Bedienungsanleitung & README

Dieses Repository enthält Code, Skripte und Ergebnisse meiner Arbeit zur **semantischen Segmentierung** (U‑Net, PyTorch) für Straßenszenen mit realen (KITTI) und synthetischen (CARLA) Daten.

> **Ziel:** Reproduzierbare Trainings/Evals für drei Setups
>
> 1. **KITTI-only** (19 Klassen)
> 2. **KITTI-Bilder + CARLA-Labelschema** (24 Klassen)
> 3. **Synthetisches CARLA** (24 Klassen)
> 4. **Mixed (KITTI+CARLA)** (24 Klassen)

---

## Verzeichnisstruktur (Repo-Root)

```
.
├─ My-Code/                     # Trainings-, Eval-, Plot- und Hilfsskripte (Python)
│  ├─ Training/                 # Train-Skripte für KITTI, CARLA, Mixed
│  ├─ Evaluation/               # Eval-/Visualisierungs-Skripte
│  ├─ Loss-Kurve-Plotten/       # Loss-Kurven aus CSV plotten
│  ├─ Datasets-vorbereitung/    # Maskenkonvertierung (KITTI→CARLA), Datensatz-Helper
│  ├─ unet/                     # U-Net Architektur (Model/Parts)
│  └─ utils/                    # Data-Loading, Dice-Score, Utils
│
├─ Carla-ID-masks/              # Beispiel-/fertige CARLA-ID-Masken (PNG)
├─ Kitti-2-Carla-masks/         # Beispiel-/fertige KITTI→CARLA-ID-Masken (PNG)
├─ Eval-Results/                # Beispiel-Evals/Visualisierungen (PNG)
├─ Loss-Kurven/                 # Beispiel-Loss CSV/Plots
└─ README.md                    # Diese Datei
```

---

## Abhängigkeiten & Setup

Empfohlen: **Python 3.10–3.11**, **CUDA‑GPU** (optional aber empfohlen).
Benötigte Pakete (ohne explizite `requirements.txt`):

```bash
# Neue Umgebung erstellen (Beispiel mit Conda)
conda create -n thesis-seg python=3.11 -y
conda activate thesis-seg

# PyTorch installieren (wähle das passende CUDA-Whl ggf. von pytorch.org)
pip install torch torchvision torchaudio

# Allgemeine Abhängigkeiten
pip install numpy pillow tqdm matplotlib pandas
```

> **Hinweis:** Für das Plotten wird `matplotlib`, für CSV‑Handling `pandas` verwendet.
> Optional: `tensorboard` für Live‑Monitoring, `scikit-learn` falls Du zusätzliche Metriken willst.

---

## Wichtiger Laufzeithinweis (Imports)

Die Skripte in `My-Code/…` importieren relative Module (z. B. `from unet import UNet`, `from KittiCarlaDataset import KittiCarlaDataset`).
**Bitte immer aus dem Ordner ********`My-Code/`******** heraus starten** oder die `PYTHONPATH` setzen:

```bash
cd My-Code
# (optional) export PYTHONPATH=.
```

---

## Daten vorbereiten

### 1) KITTI → CARLA‑ID‑Masken erzeugen (falls neu)

Im Repo liegen bereits Beispiele unter `Kitti-2-Carla-masks/`. Um selbst neu zu generieren:

* Skript: `My-Code/Datasets-vorbereitung/kitti-2-Carla-masks.py`
* Im Skript die Pfade oben anpassen (`input_dir`, `output_dir`).
* Starten (aus `My-Code/`):

```bash
python Datasets-vorbereitung/kitti-2-Carla-masks.py
```

### 2) CARLA RGB‑Masken → ID‑Masken (pro Town)

Beispiele liegen unter `Carla-ID-masks/`. Für eigene Daten:

* Skript: `My-Code/Datasets-vorbereitung/Carla-masks-ID-umwandeln.py`
* `towns = { … }` und `output_root` im Skript an Deine Pfade anpassen.
* Starten:

```bash
python Datasets-vorbereitung/Carla-masks-ID-umwandeln.py
```

> **Tipp:** Achte darauf, dass Bild‑ und Maskenauflösungen zusammenpassen (Breite/Höhe identisch).

---

## Training

Die Trainingsskripte verwenden aktuell **Konstanten im Skriptkopf** (keine CLI‑Args).
**Passe vor dem Start die Pfade** (z. B. `IMAGE_DIR`, `MASK_DIR`, `CHECKPOINT_PATH`, ggf. `CARLA_IMAGE_DIRS`) **und Hyperparameter** (`EPOCHS`, `BATCH_SIZE`, `LR`, `NUM_CLASSES`) **an**.

> **Speicherort‑Empfehlung:** Setze `CHECKPOINT_PATH` und ggf. `HISTORY_CSV` auf einen Pfad **im Repo**, z. B. `./checkpoints/...` und `./Loss-Kurven/...`.

### A) KITTI‑only (19 Klassen)

* Skript: `My-Code/Training/train-kitti-2-mit-diceloss.py`
* Wichtige Konstanten: `NUM_CLASSES = 19`, `IMAGE_DIR`, `MASK_DIR`, `CHECKPOINT_PATH`
* Start:

```bash
cd My-Code
python Training/train-kitti-2-mit-diceloss.py
```

### B) KITTI‑Bilder + CARLA‑IDs (24 Klassen)

* Skript: `My-Code/Training/train-Kitti-mit-Carla-masks.py`
* Nutzt KITTI‑RGB‑Bilder + konvertierte **CARLA‑ID‑Masken** aus `Kitti-2-Carla-masks/`
* Konstanten: `NUM_CLASSES = 24`, `IMAGE_DIR`, `MASK_DIR`, `CHECKPOINT_PATH`, `LAST_PATH`, `HISTORY_CSV`
* Start:

```bash
cd My-Code
python Training/train-Kitti-mit-Carla-masks.py
```

### C) CARLA‑synthetisch (24 Klassen)

* Skript: `My-Code/Training/train_synthetic_carla.py`
* Konstanten: `NUM_CLASSES = 24`, `MASK_ROOT` (CARLA‑ID‑Masken), Pfade zu den CARLA‑RGB‑Bildern, `CHECKPOINT_PATH`, `HISTORY_CSV`
* Start:

```bash
cd My-Code
python Training/train_synthetic_carla.py
```

### D) Mixed (KITTI + CARLA) (24 Klassen)

* Skript: `My-Code/Training/train-mixed.py`
* Konstanten: u. a. `KITTI_IMAGE_DIR`, `KITTI_MASK_DIR`, `CARLA_IMAGE_DIR = {TownXX: images_rgb}`, `CARLA_MASK_DIR`, `CHECKPOINT_DIR`, `EPOCHS`, `BATCH_SIZE`, `LR`
* Start:

```bash
cd My-Code
python Training/train-mixed.py
```

> **GPU‑Tipp:** Bei CUDA‑Out‑of‑Memory `BATCH_SIZE` reduzieren und/oder Bildgröße verkleinern (falls im Skript vorgesehen).

---

## Evaluation & Visualisierung

### 1) Einzel‑Setups

* **KITTI + CARLA‑Masken**: `My-Code/Evaluation/eval-Kitti-mit-Carla-masks.py`
  Konstanten: `CHECKPOINT_PATH`, `IMAGE_DIR`, `MASK_DIR`, `SAVE_DIR`, `NUM_CLASSES = 24`

* **CARLA**: `My-Code/Evaluation/eval_synthetic_Carla.py`
  Konstanten: `CHECKPOINT_PATH`, `IMAGE_DIR`, `MASK_DIR`, `SAVE_DIR`

* **Mixed**: `My-Code/Evaluation/eval-mixed.py`
  Konstanten: KITTI‑ & CARLA‑Pfade, `CHECKPOINT_PATH`, `SAVE_DIR`

Start (Beispiel):

```bash
cd My-Code
python Evaluation/eval-Kitti-mit-Carla-masks.py
```

### 2) Gemeinsame Auswertung aller drei Modelle

* Skript: `My-Code/Evaluation/evaluation-alle.py`
* Trägt drei Modellpfade in `MODELS = { … }` ein und erzeugt Visualisierungen in `OUT_DIR` (z. B. `../Eval-Results/`).

---

## Loss‑Kurven plotten

Die Plot‑Skripte akzeptieren **optional** den Pfad zur `loss_history.csv` als CLI‑Argument (sonst Default aus dem Skript). Beispiele:

```bash
cd My-Code
# Mixed‑Training
python Loss-Kurve-Plotten/plot-Mixed-loss.py ../Loss-Kurven/mixed_loss_history.csv

# KITTI
python Loss-Kurve-Plotten/plot-Kitti-loss.py ../Loss-Kurven/kitti_loss_history.csv

# CARLA
python Loss-Kurve-Plotten/plot-Carla-loss.py ../Loss-Kurven/carla_loss_history.csv
```

> Die Plots (PNG) werden im gleichen Ordner wie die CSV gespeichert (oder in den im Skript definierten Output‑Pfad).

---

## Ergebnisse & Beispiele

* Beispiel‑Visualisierungen liegen unter `Eval-Results/` (PNG).
* Loss‑CSV/Plots liegen unter `Loss-Kurven/`.

> Eigene Ergebnisse bitte **nicht** in die vorhandenen Beispielordner schreiben, sondern neue Unterordner anlegen (z. B. `Eval-Results/exp_YYYYMMDD/`).

---

## Häufige Stolpersteine (Troubleshooting)

* **ImportError (Module nicht gefunden):** Aus `My-Code/` starten oder `PYTHONPATH=.` setzen.
* **Pfadfehler:** In Trainings/Eval‑Skripten die Konstanten oben anpassen (Bilder, Masken, Checkpoints, CSV).
* **CUDA Out Of Memory:** `BATCH_SIZE` reduzieren, evtl. Bildgröße/Transforms anpassen.
* **GitHub 100 MB‑Limit:** Große `.pth`‑Modelle **nicht** committen; lieber lokal speichern oder Git LFS verwenden.

---

```bash
```
