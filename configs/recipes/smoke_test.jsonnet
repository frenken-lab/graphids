{
  recipe: {
    name: 'smoke_test',
    description: 'Medium-fidelity smoke -- all stages, all fusions, 50 epochs, hcrl_sa',
  },
  seeds: [99],
  selection: {
    datasets: ['hcrl_sa'],
    model_families: ['unsupervised', 'supervised', 'fusion'],
    scales: ['small'],
    stages: {
      unsupervised: ['autoencoder'],
      supervised: ['supervised'],
      fusion: ['fusion'],
    },
    fusion_methods: ['bandit', 'dqn', 'mlp', 'weighted_avg'],
  },
  trainer_overrides: {
    'trainer.max_epochs': 50,
  },
  resource_overrides: {
    time: '1:00:00',
    partition: 'gpudebug',
  },
}
