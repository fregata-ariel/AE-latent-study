"""Default configuration for lattice modular autoencoder experiments."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.seed = 42

    # --- Data ---
    config.data = ml_collections.ConfigDict()
    config.data.data_type = 'lattice'           # 'torus' | 'lattice'
    config.data.signal_length = 100             # number of t sample points
    config.data.n_train = 5000                  # 2D parameter → need more data
    config.data.n_val = 1000
    config.data.n_test = 1000
    config.data.noise_std = 0.0                 # additive Gaussian noise std

    # Lattice-specific parameters
    config.data.lattice_K = 10                  # theta function sum range |m|,|n| ≤ K
    config.data.lattice_t_min = 0.5             # theta function t range start
    config.data.lattice_t_max = 5.0             # theta function t range end
    config.data.lattice_y_max = 3.0             # Im(τ) upper bound (cusp control)
    config.data.lattice_sample_region = 'fundamental_domain'  # 'fundamental_domain' | 'halfplane'

    # --- Model ---
    config.model = ml_collections.ConfigDict()
    config.model.latent_type = 'standard'       # 'standard' | 'halfplane'
    config.model.latent_dim = 2                 # R^d dim for standard
    config.model.encoder_hidden = (256, 128, 64)
    config.model.decoder_hidden = (64, 128, 256)
    config.model.activation = 'relu'            # 'relu' | 'tanh' | 'gelu'

    # --- Training ---
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 128
    config.train.num_epochs = 300               # longer for 2D parameter space
    config.train.learning_rate = 1e-3
    config.train.weight_decay = 0.0
    config.train.lr_schedule = 'constant'       # 'constant' | 'cosine'
    config.train.patience = 30                  # early stopping patience
    config.train.log_every = 10                 # log metrics every N epochs

    # --- Checkpoint ---
    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.dir = 'checkpoints/'
    config.checkpoint.max_to_keep = 3

    # --- Evaluation ---
    config.eval = ml_collections.ConfigDict()
    config.eval.output_dir = 'results/'
    config.eval.n_interpolation = 50
    config.eval.use_umap = False

    return config
