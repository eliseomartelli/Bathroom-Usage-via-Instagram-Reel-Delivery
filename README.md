# Inferring Bathroom Usage via Instagram Reel Delivery

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](main.pdf)
[![Analysis](https://img.shields.io/badge/Analysis-Python-blue)](test.py)

This repository contains the research, data extraction pipeline, and
statistical models for inferring the timing of human gastrointestinal routines
using passive digital footprints. Specifically, it explores the use of
**Instagram Reel delivery data** as a proxy for bathroom usage patterns through
Hidden Markov Models (HMM) and Density-Based Spatial Clustering (DBSCAN).

**SATIRE WARNING**: This project is a tongue-in-cheek exploration of the
intersection between digital behavior and human physiology. It is not intended
for actual medical inference or diagnosis. The methodologies and conclusions
are presented with a humorous lens and should be interpreted as a playful
thought experiment rather than rigorous scientific analysis.

## Overview

The ubiquity of smartphones has fundamentally altered human colonic habits.
This project demonstrates a statistically significant correlation between
concentrated meme sharing ("Doomscrolling") and periods of bowel evacuation. By
analyzing the JSON archive of a user’s social media account, we
probabilistically reconstruct physical realities from virtual metadata.

### Key Features:
- **Digital Phenotyping**: Moment-by-moment quantification of human behavior in
situ.
- **Burst Detection**: Implementation of `sklearn.cluster.DBSCAN` to identify
dense temporal clusters of media shares.
- **Chronobiological Priors**: Integration of colonic circadian rhythms
(awakening surges and gastrocolic reflexes) into a Bayesian inference engine.
- **Automated Visualization**: Generation of log-log IAT distributions,
spatiotemporal heatmaps, and burst topography charts.

## Repository Structure

- `main.tex`: The full LaTeX manuscript (Abstract, Methodology, Experimental
Results, etc.).
- `test.py`: The Python data processing script utilizing `pandas`, `numpy`, and
`scikit-learn`.
- `hourly_Subject_*.png`: Temporal distribution of incoming reels for
anonymized subjects.
- `timeline_Subject_*.png`: Inferred activity timelines.
- `iat_loglog.png`: Inter-arrival time distribution validating the Barabási
heavy-tail hypothesis.
- `heatmap_dow_hour.png`: Spatiotemporal aggregate of inferred events.
- `burst_density_bubble.png`: Topographical map of burst volume vs. duration.

## Getting Started

### Prerequisites
- Python 3.8+
- LaTeX distribution (e.g., TeX Live)
- Required Python packages:
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install -r requirements.txt
  ```

### Data Acquisition
1. Request a data export from Instagram (Meta Accounts Center).
2. Select **"Messages"** as the information type.
3. Explicitly set the format to **JSON**.
4. Once received, extract the archive and point the `INBOX_PATH` variable in
   `test.py` to your `messages/inbox/` directory.

### Running the Analysis
To process the data and generate all visualizations:
```bash
python3 test.py
```

To compile the manuscript:
```bash
pdflatex main.tex
```

*Note: This project is intended for research purposes into digital phenotyping
and temporal patterns. All data used in this repository has been anonymized.*
