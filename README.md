## Exputils: Experiment Utilities for Expediency

Convenient scripts and functions that are commonly used in code for running experiments, specifically machine learning experiments.
This project is intended to expedite the research coding process.

The main components are:

- io: nest namespaces in argparse, ease of saving JSONs, default arg parser, logging, and eventually profiling helpers (e.g. runtime at least)
- data: kfold cv wrapper and abstract data class
- visuals: Visualization scripts for common plots via pyplot

### Design Principles

- Keep it simple
    - and functional
- Write once
    - reduce redundant code and make it so common code does not need be rewritten for the same functionality.
- Modularity
    - Keep as modular as possible such that each unit may be removed, replaced, or taken and plugged into a different system.
- Efficiency
    - efficient coding and execution

### Features

- exputils.data
    - Confusion matrix: streamlines obtaining, modifying, combining, deriving summary measures, saving/loading, and visualizing of confusion matrices.
    - exputils.data.handlers
- exputils.io
- exputils.ml
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

Docstr uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Docstr's version will remain < 1.0.0 until adequate unit test coverage exists.

### License

The exputils project is licensed under the MIT license.
The license is provided in LICENSE.txt
