## Data

Useful code for handling data.

### Features

- Label encoder that extends the scikit learn encoder as a call object with state that carries the encoder and its labels with it for convenience.
- Confusion matrix code that calculates common metrics from the confusion matrix itself, rather than the from the raw data as scikit-learn measures do.
    Also has convenient I/O functions and heatmap visualization.

#### TODO

- extend the Nominal Label Encoder to better  handle unlabeled data, group labels, and heirarchy/structured labels.
- add generic data loader classes to streamline data loading in ML problems.
- add image augmentation data loaders which append to data loaders and
    - try to make it such that these functional called/appended data loaders are efficient in their respective language/lib (python, numpy, pytorch, tf, jax).
