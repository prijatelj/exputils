"""Example showing how to use exputils.io's initial argparser."""
import logging

from exputils import io


def add_example_args(parser):
    example = parser.add_argument_group(
        'example',
        'Example args grouped together from adding custom args.',
    )

    example.add_argument(
        '--dir_path',
        help=' '.join([
            'The filepath to the directory containing either the separate',
            'results directories of a kfold experiment, or the root directory',
            'containing all kfold experiments to be loaded.',
        ]),
        default='./',
        dest='example.dir_path',
    )

    example.add_argument(
        '--summary_name',
        help=' '.join([
            'The relative filepath from the eval fold directory to the',
            'summary JSON file.',
        ]),
        default='summary.json',
        dest='example.summary_name',
    )


def add_custom_args(parser):
    add_example_args(parser)

    parser.add_argument(
        '--save_labels',
        action='store_true',
        help='Save the labels in the JSON of the predictions.'
    )

    parser.add_argument(
        '--pie_cake',
        action='store_true',
        help='Example of how NestedNamespaces recurse based on given dest',
        dest='so.much.pie.cake',
    )


if __name__ == '__main__':
    args = io.parse_args(custom_args=add_custom_args)

    logging.info('It ran.')
