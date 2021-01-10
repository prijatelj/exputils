## Experiment Utils

Convenient scripts and functions that are commonly used in code for running experiments, specifically machine learning experiments.

The main components are:

- io: nest namespaces in argparse, ease of saving JSONs, default arg parser, logging, and eventually profiling helpers (e.g. runtime at least)
- data: kfold cv wrapper and abstract data class
- visuals: Visualization scripts for common plots via pyplot

### TODO
+ Add general configuration file handling and connect to the default initial argparser.
    - support yaml and JSON
+ Improve the NestedNamespace creation process so it is simpler and streamlined.
    - make it so the user does not need to specify dest, as it is duplicative
    - the nested namespaces have args mutually exclusive only to their current namespace (node in the namespace tree), allowing separate namespaces to use the same argument name as another.
        + This requires the command line interface to provide the argparse namespace identifier when the user supplies arguments. (i.e. long arg format)
        e.g. path from top level namespace to the node of the nested namespace whose args are being modified.
+ Perhaps here is not best, but being able to turn any table (csv, tsv) into a LaTeX table would be wonderful for expediency of putting it into a paper.
+ Consider a confusion tensor when addressing multiple random variables, e.g. a tensor with n+1 dimensions: actual/ground truth, predictor 1, predictor 2, ... predictor n.
