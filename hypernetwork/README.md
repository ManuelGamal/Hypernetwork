# Hypernetwork — Networks That Generate Networks

A minimal, well-documented implementation of a **hypernetwork**: a neural network that generates the weights of another neural network, rather than learning them directly. This project illustrates how parameter prediction can be used as a path toward more compute-efficient training.

---

## Concept

In a standard neural network, weights are learned via backpropagation. In a **hypernetwork**:

```
Input (task embedding / context)
        ↓
  [ HyperNetwork ]   ← trains via backprop
        ↓
  Generated Weights
        ↓
  [ Target Network ] ← weights are injected, not trained directly
        ↓
     Prediction
```

The hypernetwork learns a *mapping from context → weights*, so a single hypernetwork can parameterize an entire family of target networks.

---

## Project Structure

```
hypernetwork/
├── src/
│   ├── hypernetwork/
│   │   ├── __init__.py
│   │   └── model.py          # HyperNetwork class
│   ├── target_network/
│   │   ├── __init__.py
│   │   └── model.py          # TargetNetwork (weights injected externally)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data.py           # Synthetic task generators
│   │   └── training.py       # Training loop
│   └── visualization/
│       ├── __init__.py
│       └── plot.py           # Weight distribution & loss plots
├── notebooks/
│   └── demo.ipynb            # End-to-end walkthrough
├── tests/
│   ├── test_hypernetwork.py
│   └── test_target_network.py
├── train.py                  # Main training script
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt
python train.py
```

Or run the notebook for an interactive walkthrough:

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Experiment: Sine Wave Family

The demo trains a hypernetwork to generate weights for a small MLP that fits **sine waves of varying frequency and phase** — without retraining the target network for each wave. The hypernetwork receives `(frequency, phase)` as input and outputs all the target network's weights in one forward pass.

This demonstrates:
- **Weight prediction**: the hypernetwork outputs ~200 floats (the target network's full parameter vector)
- **Generalization**: after training, the hypernetwork generalizes to unseen (freq, phase) combinations
- **Efficiency**: one shared hypernetwork replaces N independently trained target networks

---

## Key Ideas Illustrated

| Concept | Where |
|---|---|
| Weight generation | `src/hypernetwork/model.py` |
| Weight injection | `src/target_network/model.py` |
| Meta-task sampling | `src/utils/data.py` |
| End-to-end training | `src/utils/training.py` |
| Loss & weight plots | `src/visualization/plot.py` |

---

## References

- Ha, D., Dai, A., & Le, Q. V. (2016). *HyperNetworks*. arXiv:1609.09106
- Oswald et al. (2020). *Continual learning with hypernetworks*. ICLR 2020
- Metz et al. (2022). *Learned optimizers*. NeurIPS 2022
