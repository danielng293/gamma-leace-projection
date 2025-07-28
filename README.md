# Gamma-Controlled-LEACE
Explored the fairness–accuracy trade-off in projection based machine learning algorithm by proposing γ-controlled LEACE, a refined version of LEACE (Least Squares Concept Erasure). Implemented synthetic data generation, custom ML pipelines, and Pareto front analysis in Python.
---
## Project Structure
├── Bonus Study/ # Additional experiments on varied datasets using gamma-controlled LEACE
│ └── different_datasets.py
├── Dataset/ # Synthetic dataset generation
│ └── synthetic_generator.py
├── LEACE/ # Gamma-controlled LEACE implementation
│ ├── gamma_controlled.py
│ └── whitening_matrix.py
├── Pareto Front/ # Main experiment and trade-off visualizations
│ ├── main_experiment.py
│ └── pareto_front.py
├── Report/ # Final paper
│ └── paper.pdf
├── requirements.txt # Python dependencies
└── README.md # Project overview (this file)
---
## How to run
```bash
pip install -r requirements.txt
python "Pareto Front/main_experiment.py"
python "Pareto Front/pareto_front.py"
```
---
## What is a γ-controlled LEACE?
LEACE (Least Squares Concept Erasure) is a projection based method proposed by Belrose et al.(2024) that removes linear correlations between sensitive attribute and features in the data to achieve a 'fairness' in machine leanring.

This project introduces γ-controlled LEACE, a modification that allows a partial erasure of sensitive information, by adjusting the γ parameter in the algorithm. By analysing varying level of fairness and accuracy for different γ values, we can trace the trade-off between fairness and accuracy.

For more detailed information and definitions, please refer to the paper in the `Report` folder.

