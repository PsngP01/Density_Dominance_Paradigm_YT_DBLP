# Density-Dominance-Paradigim_Youtube
Python implementation of the Influencer Matrix and IC diffusion simulation from 'The Density-Dominance Paradigm' (IEEE TNSE).
# The Density-Dominance Paradigm: Influencer Matrix Framework

This repository contains the Python code for extracting topological fingerprints, discovering structural archetypes (PCA + K-Means), and running Independent Cascade (IC) diffusion simulations as described in our IEEE manuscript.

## Prerequisites
To run this code, you will need the following libraries:
`pip install networkx pandas numpy matplotlib seaborn plotly scikit-learn scipy tqdm joblib`

## Dataset Download
Because of GitHub's file size limits, the SNAP datasets are not included in this repository. 
1. Download the YouTube graph and community data from the [Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/data/com-Youtube.html).
2. Create a folder named `data` in the same directory as the script.
3. Place `com-youtube.ungraph.txt` and `com-youtube.all.cmty.txt` into the `data` folder.

## Running the Code
Simply execute the script:
`python influencer_matrix_youtube.py`
