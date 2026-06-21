# DR-ENb0 — Diabetic Retinopathy Severity Grading

Research pipeline for diabetic retinopathy severity grading on APTOS 2019 fundus images using EfficientNet-B0 with explainable AI (Grad-

## Getting Started

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yassyass2/DR-ENb0.git
   cd DR-ENb0
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

## Project Structure

```
├── capstone-project/
│   ├── models/          # Model notebooks (EfficientNet-B0 variants)
│   └── src/             # Source code / pipeline modules
├── pyproject.toml
└── uv.lock
```

## Running the Preprocessing Pipeline

```bash
uv run main.py
```

## Models

The model notebooks are located in the `capstone-project/models/` directory. Each notebook covers a specific model variant or experiment (e.g. baseline, SMOTE, focal loss).

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) — fundus images graded 0–4 by DR severity.

## Contributors

- Adel
- Mark
- Musab
- Ozeir
- Yassine
