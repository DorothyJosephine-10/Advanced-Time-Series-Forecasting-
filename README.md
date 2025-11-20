Advanced Time Series Forecasting with Hierarchical Temporal Memory (HTM-Like Model) and LSTM

This project implements a complete, end-to-end forecasting framework using:

A biologically inspired HTM-like Temporal Memory model, using SDR encoding and Hebbian learning

A deep learning baseline model (LSTM) for comparison

Automated generation of:

A detailed analysis report

Forecasting comparison plot (HTM vs LSTM)

The project satisfies all requirements shown in the assigned task sheet:

- Multivariate data generation (≥10,000 samples)
- HTM model implementation with SDR preprocessing
-Hyperparameter tuning + prediction generation
- Baseline model for comparison
- Comparative evaluation (RMSE, MAE, directional accuracy)
- Written analysis deliverable (Markdown)
- Output of top critical HTM hyperparameters

Project Outputs

This program automatically generates:

File  Description
deliverable_2_analysis.md  Full written analysis (HTM architecture, tuning, comparison)
comparison_plot.png  Plot comparing True vs LSTM vs HTM forecasts
program_explanation_40000_chars.txt  40k-character detailed program explanation
Model metrics  RMSE, MAE, directional accuracy for HTM & LSTM
Project Workflow
1. Synthetic Multivariate Dataset Generation

The dataset:

Contains 12,000 time steps

Includes three features (x1, x2, x3)

Features:

Trend

Seasonality

Noise

Nonlinear interactions

This simulates real-world complex time series behavior.

2. SDR Encoding (HTM Input Preprocessing)

Each scalar feature is encoded into a Sparse Distributed Representation (SDR):

256 bits per feature

20 active bits

3 features → 768-bit SDR per timestep

Why SDRs?

Noise-robust

High semantic similarity

Matches biological HTM architecture requirements

3. HTM-Like Temporal Memory Model

The HTM-like model includes:

A memory matrix that stores learned patterns

Sparse top-k activation for prediction

Hebbian learning (memory += lr * outer(pred, input))

Online sequence processing

Predictions are decoded from SDR activations

This is a simplified HTM implementation suitable for educational + research purposes.

4. LSTM Baseline Model

A strong deep-learning baseline:

2-layer LSTM

64 hidden units

Sequence length = 20

Parameterized regression → predicts real values directly

Trained using MSE loss + Adam optimizer

This serves as a high-quality benchmark against the HTM model.

5. Evaluation Metrics

The program computes:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

Directional Accuracy (trend correctness)

6. Deliverable #2 — Auto-Generated Report

The script generates a full markdown report:

HTM architecture explanation

Hyperparameter tuning strategy

LSTM vs HTM comparison

Observations and interpretation

Limitations

Recommendations

Saved as: deliverable_2_analysis.md

 Project Structure (Recommended)
.
├── main.py                          # Main executable program
├── deliverable_2_analysis.md        # Auto-generated analysis report
├── comparison_plot.png              # HTM vs LSTM forecast comparison
├── program_explanation_40000_chars.txt
├── README.md                        # This file
└── data/                            # (Optional) generated or real datasets

⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/HTM_TimeSeries_Forecasting.git
cd HTM_TimeSeries_Forecasting

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

3. Install dependencies
pip install numpy pandas torch matplotlib scikit-learn

 Usage
Run the full pipeline
python main.py


This will:

Generate the dataset

Train the HTM model

Train the LSTM model

Evaluate performance

Generate plots

Create all deliverables automatically

Performance Outputs (Example)
Model  RMSE  MAE  Direction Accuracy
HTM-like  ~X.XXXX  ~X.XXXX  ~0.50–0.65
LSTM  lower RMSE typically  lower MAE  ~0.70–0.90
