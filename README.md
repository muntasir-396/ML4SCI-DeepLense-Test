# ML4SCI DeepLense: GSoC Evaluation Tests

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Applicant:** Muntasir Mamun  
**Organization:** Machine Learning for Science (ML4SCI)  
**Project:** DeepLense (Deep Learning for Strong Gravitational Lensing)  

---

## 🌌 Overview

Strong gravitational lensing occurs when massive foreground galaxies bend light rays originating from background galaxies. Dark matter structures—specifically smooth halos, subhalos (cold dark matter clumps), and vortex substructures—leave subtle morphological imprints on the Einstein rings and arcs.

This repository contains the complete implementation and evaluation benchmarks for prospective GSoC contributors to the **ML4SCI DeepLense** program:
1. **Common Test I:** Multi-class classification distinguishing smooth lenses, subhalos, and vortex substructures.
2. **Specific Test II:** Agentic AI workflow for autonomous gravitational lensing simulation orchestration.

---

## 📊 Common Test I: Multi-Class Classification

### 1. Dataset & Preprocessing
* **Classes:**
  - `no_sub`: Smooth lensing without substructure.
  - `subhalo`: Gravitational lenses with cold dark matter subhalos.
  - `vortex`: Gravitational lenses with vortex substructures.
* **Split:** Official **90:10 stratified train-validation split** ensuring class distribution parity.
* **Augmentation:** Rotational and reflective invariance transformations (Random Horizontal/Vertical Flips, 90° Rotations) reflecting physical symmetry.

### 2. Architecture & Training Details
* **Backbone:** ResNet-50 transfer learning pretrained on ImageNet.
* **Custom Head:** Feature projection to 256 dimensions with Batch Normalization, ReLU, Dropout ($p=0.3$), and a 3-class linear classifier.
* **Optimization:** AdamW optimizer ($\text{lr} = 10^{-4}, \text{weight\_decay} = 10^{-2}$) with Cosine Annealing learning rate schedule.
* **Loss Function:** Categorical Cross-Entropy Loss.

### 3. Evaluation Metrics
Evaluated on the 10% hold-out validation set:
* **One-vs-Rest (OvR) ROC Curves** plotted for all three classes.
* **Area Under the ROC Curve (AUC)** computed per-class and Macro-averaged.

---

## 🤖 Specific Test II: Agentic AI Workflow

Implements an autonomous agent using **Pydantic** structured models and LLM tool calling to interface with `DeepLenseSim`:
* **Pydantic Validation:** Strongly-typed simulation parameters (`substructure_type`, `model_config`, `redshift_lens`, `redshift_source`, `num_images`).
* **Tool Calling Execution:** Safe invocation of the simulation environment.
* **Human-in-the-Loop:** Interactive clarification loops for ambiguous scientific prompts.

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/muntasir-396/ML4SCI-DeepLense-GSoC-Test.git
cd ML4SCI-DeepLense-GSoC-Test
pip install -r requirements.txt
```

### Run Locally or in Jupyter
```bash
jupyter notebook Common_Test_I_Classification.ipynb
```
