# 5HT7a-ligand-binding-demo
A small, non-proprietary demo showing how to build a ligand-based model for predicting 5-HT7A receptor binding affinity (pKi). Includes RDKit-based featurization, physicochemical descriptor generation, and a simple machine-learning model to illustrate my coding style and interest in computational drug discovery.
# 5-HT7A Ligand Binding Demo

This repository contains a small, non-proprietary demo illustrating how to build a
simple QSAR-style model to predict binding affinity (pKi) of ligands for the
5-HT7A receptor using cheminformatics features.

The workflow:
- Takes example ligands as SMILES strings with mock pKi values.
- Uses RDKit to generate Morgan fingerprints and physicochemical descriptors.
- Trains a RandomForestRegressor to map features → pKi.
- Evaluates performance with basic regression metrics.

This is a toy project meant to showcase my coding style and my interest in
drug discovery, ligand-based modeling, and serotonin 5-HT7A medicinal chemistry.
