# Classification of ceramic samples by chemical element concentrations

This repository contains the code used for the experiments described in
the paper "Supervised learning algorithms as a tool for archaeology:
classification of ceramic samples described by chemical element
concentrations", by G. Ruschioni, D. Malchiodi, A. M. Zanaboni and
L. Bonizzoni1, currently submitted to Journal of Archaeological Science.

Python 3.8 is used, as well as some standard libraries described in
`requirements.txt`. The same file can be used in order to create a pip
virtual environment to be used for all experiments.

Data, available upon request to the authors, should be placed in the `data`
folder.

The `experiments/JAS` folder contains all scripts and notebooks to be run
to replicate experiments:
- [`experiments-with-dim-reduction.py`](experiments/JAS/experiments-with-dim-reduction.py) trains all classifiers described in
  Sections 4 and 5 of the paper;
- [`generate-heatmaps.ipynb`](experiments/JAS/generate-heatmaps.ipynb)
  generates the heatmaps of Figures 3--5;
- [`show-dt.ipynb`](experiments/JAS/show-dt.ipynb) graphically depicts the
  learnt Decision Trees, as described in Section 5.

The `models` folder contains the serialization of all learnt classifiers,
as well as some CSV used by the notebooks. Its contents are organized in
three sub-folders `all_measures`, `frag_nosample` and `frag_sample-2`,
referring to the three groups of experiments described, respectively, as
"type 1", "type 2" and "type 3" experiments in the paper.