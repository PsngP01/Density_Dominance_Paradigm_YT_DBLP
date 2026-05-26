# Density-Dominance-Paradigim_Youtube_DBLP
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


# The Density-Dominance Paradigm: Influencer Matrix Framework

This repository contains the Python code for extracting topological fingerprints, discovering structural archetypes (PCA + K-Means), and running Independent Cascade (IC) diffusion simulations across two cross-domain topologies (YouTube and DBLP) as described in our IEEE manuscript.

## Prerequisites
To run this code, you will need the following libraries:
`pip install networkx pandas numpy matplotlib seaborn plotly scikit-learn scipy tqdm joblib`

## Dataset Download
Because of GitHub's file size limits, the SNAP datasets are not included in this repository. 
1. Create a folder named `data` in the same directory as the scripts.
2. Download the datasets from the [Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/data/):
   - **YouTube:** `com-youtube.ungraph.txt` and `com-youtube.all.cmty.txt`
   - **DBLP:** `com-dblp.ungraph.txt` and `com-dblp.all.cmty.txt`
3. Place all four text files into the `data` folder.

## Running the Code
To run the social network analysis (YouTube):
`python influencer_matrix_youtube.py`

To run the academic collaboration analysis (DBLP):
`python influencer_matrix_dblp.py`
