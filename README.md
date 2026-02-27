# TabICL-Survival: A Tabular Foundation Model for Survival In-Context Learning

This repo is an adaptation of ["TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"](https://arxiv.org/pdf/2502.05564) for survival analysis. It is trained using Cox negative log likelihood.

## Installation

Option 1: Installing `tabicl-survival` from PyPI

```bash
pip install tabicl-survival
```

Option 2: Installing `tabicl-survival` from the local clone

```bash
cd tabicl
pip install -e .
```

Option 3: Installing `tabicl-survival` directly from the git remote

```bash
pip install git+https://github.com/taltstidl/tabicl-survival.git
```

## Basic Usage

```python
from tabicl import TabICLSurver

surv = TabICLSurver()
surv.fit(X_train, y_train)  # this is cheap
surv.predict(X_test)  # in-context learning happens here
```

The code above will automatically download the pre-trained checkpoint (~325MB) from Hugging Face Hub on first use and choose a GPU if available. It supports datasets with up to 1,024 samples and 100 features at the moment.

## Contributors

For the original TabICL implementation
- [Jingang Qu](https://github.com/jingangQu)
- [David Holzmüller](https://github.com/dholzmueller)
- [Marine Le Morvan](https://github.com/marineLM)

For the adapted TabICL-Surival implementation
- [Thomas Altstidl](https://github.com/taltstidl)
