## Exputils: Experiment Utilities for Expediency

A toolkit of convenient scripts and functions that are commonly used in my (Derek Prijatelj's) code for running experiments, specifically machine learning experiments.
This project is intended to expedite the research coding process by serving as a toolkit.
Though if a tool is developed enough, it may be split up into its own package.

The main components are:

- io: nest namespaces in argparse, default arg parser, ease of saving JSONs, logging, and eventually profiling helpers (e.g. runtime at least)
    - [docstr](https://github.com/prijatelj/docstr) and soon to be [typing-to-configargparse](https://github.com/prijatelj/typing-to-configargparse) as well, provide the argparse tools now and further development is done through them.
- data: kfold cv wrapper and abstract data class
- visuals: Visualization scripts for common plots via pyplot

### Design Principles

1. Write once
    - reduce redundant code and the need for rewriting code for the same functionality.
2. Keep it simple
    - and functional
3. Modularity
    - Keep as modular as possible such that each unit may be removed, replaced, or taken and plugged into a different system.
4. Efficiency
    - efficient coding and execution

### Features

- exputils.data
    - Confusion matrix: streamlines obtaining, modifying, combining, deriving summary measures, saving/loading, and visualizing of confusion matrices.
    - Ordered Confusion Tensor, really ordered confusion matrices to get top-k measures, such as top-5 accuracy, as well as get the original confusion matrix.
        Essentially, storing the order of class predictions over all samples.
    - exputils.data.handlers
- exputils.io
    - basic conveniences for input and output, mostly for creating a filepath and making a new unique name if it already exists by appending datetime to it.
- exputils.ml
    - just generic class structure, mostly for reference.
- [-TODO-]
    - exputils.profile
    - exputils.ray
    - exputils.visuals


#### TODO

+ Add general configuration file handling and connect to the default initial argparser.
    - support yaml and JSON
+ Improve the NestedNamespace creation process so it is simpler and streamlined.
    - make it so the user does not need to specify dest, as it is duplicative
    - the nested namespaces have args mutually exclusive only to their current namespace (node in the namespace tree), allowing separate namespaces to use the same argument name as another.
        + This requires the command line interface to provide the argparse namespace identifier when the user supplies arguments. (i.e. long arg format)
        e.g. path from top level namespace to the node of the nested namespace whose args are being modified.
+ Perhaps here is not best, but being able to turn any table (csv, tsv) into a LaTeX table would be wonderful for expediency of putting it into a paper.
+ Consider a confusion tensor when addressing multiple random variables, e.g. a tensor with n+1 dimensions: actual/ground truth, predictor 1, predictor 2, ... predictor n.

### Verisoning

Exputils uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Exputils's version will remain < 1.0.0 until adequate unit test coverage exists.

### License

The exputils project is licensed under the MIT license.
The license is provided in LICENSE.txt
