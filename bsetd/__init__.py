"""BSETD: Bayesian Spectral Emotion Transition Discovery.

Two-stage framework for discovering and visualizing emotion-to-emotion
relations in continuous dialogues with multi-annotator soft labels:

    Stage 1: Hierarchical Dirichlet-Multinomial empirical Bayes
             estimation of the K x K soft-label transition matrix.
    Stage 2: Symmetrized graph Laplacian spectral decomposition
             that separates inertia (low-frequency) from
             contagion/shift (high-frequency) patterns.
"""
