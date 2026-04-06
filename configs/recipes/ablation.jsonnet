{
  recipe: {
    name: 'ablation',
    description: 'Focused factorial and method ablations (claims 2, 4, 5, 6)',
  },
  sweeps: [
    // Claim 4: Loss function -- supervised stage
    {
      model_family: 'supervised',
      stage: 'supervised',
      scale: ['small', 'large'],
      model_overrides: {
        init_args: {
          loss_fn: ['ce', 'focal', 'weighted_ce'],
        },
      },
    },
    // Claim 2: Fusion method -- all 4 fusion methods
    {
      model_family: 'fusion',
      stage: 'fusion',
      scale: ['small', 'large'],
      fusion_method: ['bandit', 'dqn', 'mlp', 'weighted_avg'],
    },
    // Claim 5: Conv type -- GATv2 (default) vs GATv1 vs GPSConv
    {
      model_family: 'supervised',
      stage: 'supervised',
      scale: 'small',
      model_overrides: {
        init_args: {
          conv_type: ['gat', 'gps'],
          loss_fn: 'focal',
        },
      },
    },
    // Claim 6: Unsupervised method -- VGAE (default) vs GAE vs DGI
    // GAE = VGAE with variational=false; DGI = separate model_type
    {
      model_family: 'unsupervised',
      stage: 'autoencoder',
      scale: 'small',
      model_overrides: {
        init_args: {
          variational: false,
        },
      },
    },
    {
      model_family: 'unsupervised',
      stage: 'autoencoder',
      scale: 'small',
    },
  ],
}
