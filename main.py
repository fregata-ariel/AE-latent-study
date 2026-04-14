"""Entry point for single experiment.

Usage:
    python main.py --config=configs/default.py --workdir=runs/default
    python main.py --config=configs/t1_standard.py --workdir=runs/t1_standard
    python main.py --config=configs/t1_torus.py --workdir=runs/t1_torus

    # Override config values from command line:
    python main.py --config=configs/t1_standard.py \\
        --config.train.num_epochs=500 \\
        --config.model.latent_dim=4
"""

from absl import app, flags
from ml_collections import config_flags

from train.trainer import train_and_evaluate
from eval.analysis import run_full_evaluation

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', 'configs/default.py', 'Path to experiment config file.',
)
flags.DEFINE_string('workdir', 'runs/default', 'Working directory for outputs.')


def main(argv):
    del argv
    config = FLAGS.config
    workdir = FLAGS.workdir

    # Train
    state, history, (train_ds, val_ds, test_ds) = train_and_evaluate(config, workdir)

    # Evaluate
    run_full_evaluation(state, config, train_ds, test_ds, history, workdir)


if __name__ == '__main__':
    app.run(main)
