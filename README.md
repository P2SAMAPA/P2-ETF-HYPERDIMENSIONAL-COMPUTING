# Hyperdimensional Computing for ETFs

Implements hyperdimensional (HD) computing – a brain‑inspired computing paradigm using high‑dimensional (10,000‑dim) holographic vectors. Operations include binding (multiplication), bundling (majority sum), and permutation (circular shift). Market states are encoded as hypervectors, and the score is the similarity to a learned "up" prototype.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Quantization of returns and macro variables into levels
- Level hypervectors for each quantization bin
- Prototype vectors for "up" and "down" days
- Score = similarity(current_state, up_prototype) - similarity(current_state, down_prototype)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-hyperdimensional-computing-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → current market state resembles historically up days → potential upward move.
- Low score → resembles down days.

## Requirements

See `requirements.txt`.
