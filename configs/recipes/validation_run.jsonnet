{
  recipe: {
    name: 'validation_run',
    description: 'Post-refactor validation -- real dagster on hcrl_sa, 3 epochs',
  },
  seeds: [42],
  selection: {
    datasets: ['hcrl_sa'],
    model_families: ['unsupervised', 'supervised', 'fusion'],
    scales: ['small'],
    stages: {
      unsupervised: ['autoencoder'],
      supervised: ['supervised'],
      fusion: ['fusion'],
    },
    fusion_methods: ['bandit'],
  },
}
