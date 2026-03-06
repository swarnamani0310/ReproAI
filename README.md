# ReproAI - IVF Decision Support System

AI-powered fertility treatment optimization prototype with Digital Twin simulation.

## Features

✅ **8 Core Components:**
1. Doctor Input Interface
2. Safety Rule Engine
3. ML Prediction Model
4. Similar Patient Finder (kNN)
5. Protocol Simulation (Digital Twin)
6. Optimization Recommendation
7. Doctor Override (Doctor-in-the-Loop)
8. Clinical Summary Dashboard

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. Enter patient details (Age, BMI, AMH, FSH, AFC, PCOS, Previous IVF)
2. Click "Analyze Patient"
3. Review AI predictions and recommendations
4. Accept, modify, or reject the recommendation

## Tech Stack

- **Python** - Core language
- **Streamlit** - Web interface
- **scikit-learn** - ML models (Random Forest, kNN)
- **pandas** - Data handling
- **numpy** - Numerical computations

## Model Details

- **Pregnancy Prediction:** Random Forest Classifier
- **Similar Patients:** k-Nearest Neighbors (k=85)
- **Training Data:** 500 synthetic patient records
- **Features:** Age, BMI, AMH, FSH, AFC, PCOS, Previous IVF attempts

## Safety Rules

- AMH < 1 → Low ovarian reserve alert
- AMH > 3.5 + PCOS → High OHSS risk
- Age > 40 → Lower success probability
- BMI > 30 → Elevated BMI warning
- FSH > 10 → Reduced ovarian reserve

## Protocol Simulation

Simulates 3 treatment protocols:
- Antagonist Protocol
- Long Agonist Protocol
- Mild Stimulation

Each shows expected eggs, success rate, and OHSS risk.

## Optimization Formula

```
Score = 0.5 × Success - 0.2 × OHSS Risk - 0.15 × Cost - 0.15 × Medication
```

Recommends protocol with highest score.
