"""Default configuration for torus autoencoder experiments."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.seed = 42

    # --- Data ---
    config.data = ml_collections.ConfigDict()
    config.data.torus_dim = 1          # 1 for T^1, 2 for T^2
    config.data.signal_length = 100    # number of time steps
    config.data.n_train = 2000
    config.data.n_val = 500
    config.data.n_test = 500
    config.data.omega = 2.0            # angular frequency for T^1
    config.data.omega1 = 2.0           # first frequency for T^2
    config.data.omega2 = 3.0           # second frequency for T^2
    config.data.a1 = 1.0              # amplitude for first component (T^2)
    config.data.a2 = 0.5              # amplitude for second component (T^2)
    config.data.noise_std = 0.0        # additive Gaussian noise std
    config.data.dt = 0.1              # time step spacing

    # --- Model ---
    config.model = ml_collections.ConfigDict()
    config.model.latent_type = 'standard'  # 'standard' | 'torus' | 'vae' | 'factorized_vae'
    config.model.latent_dim = 2            # R^d dim for standard; n_angles for torus
    config.model.quotient_dim = 2          # quotient-chart dimension for factorized_vae
    config.model.gauge_dim = 4             # gauge dimension for factorized_vae
    config.model.gauge_action_type = 'affine'
    config.model.encoder_hidden = (256, 128, 64)
    config.model.decoder_hidden = (64, 128, 256)
    config.model.activation = 'relu'       # 'relu' | 'tanh' | 'gelu'
    config.model.vae_beta = 1.0            # KL weight for VAE

    # --- Training ---
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 128
    config.train.num_epochs = 200
    config.train.learning_rate = 1e-3
    config.train.weight_decay = 0.0
    config.train.lr_schedule = 'constant'  # 'constant' | 'cosine'
    config.train.patience = 20             # early stopping patience
    config.train.log_every = 10            # log metrics every N epochs
    config.train.modular_invariance_weight = 0.0
    config.train.gauge_equivariance_weight = 0.03
    config.train.decoder_equivariance_weight = 0.03
    config.train.gauge_action_reg_weight = 1e-4
    config.train.chart_preserving_weight = 0.0
    config.train.chart_preserving_n_neighbors = 8
    config.train.quotient_variance_floor_weight = 0.0
    config.train.quotient_variance_floor_target = 0.15
    config.train.quotient_spread_weight = 0.0
    config.train.quotient_min_eig_ratio_target = 0.20
    config.train.quotient_trace_cap_ratio = 1.50
    config.train.jacobian_gram_weight = 0.0
    config.train.jacobian_n_neighbors = 8
    config.train.quotient_logdet_weight = 0.0
    config.train.quotient_logdet_ratio_target = 0.10
    config.train.contrastive_local_weight = 0.0
    config.train.contrastive_n_neighbors = 8
    config.train.contrastive_temperature = 0.10
    config.train.j_rank_preserving_weight = 0.0
    config.train.j_rank_temperature = 0.10
    config.train.j_rank_min_delta = 0.10
    config.train.j_rank_n_terms = 50
    config.train.teacher_distill_weight = 0.0
    config.train.teacher_run_dir = ''
    config.train.teacher_distill_n_neighbors = 8
    config.train.teacher_distill_view = 'quotient'
    config.train.teacher_distill_loss_type = 'local_distance'

    # --- Checkpoint ---
    config.checkpoint = ml_collections.ConfigDict()
    config.checkpoint.dir = 'checkpoints/'
    config.checkpoint.max_to_keep = 3

    # --- Evaluation ---
    config.eval = ml_collections.ConfigDict()
    config.eval.output_dir = 'results/'
    config.eval.n_interpolation = 50
    config.eval.use_umap = False
    config.eval.chart_n_neighbors = 8
    config.eval.chart_max_samples = 2000
    config.eval.ph_enabled = False
    config.eval.ph_max_samples = 2000
    config.eval.ph_proj_dims = ()
    config.eval.ph_maxdim = 1
    config.eval.ph_random_projection_trials = 8
    config.eval.ph_knn_for_lid = 10
    config.eval.ph_noise_floor = 0.05
    config.eval.ph_noise_floor_mode = 'relative'

    return config
