// Ablation arm 2/3: lookup + stochastic UNK-drop during training.
// Stage-3 design-decision arm — node_ids are remapped to UNK_INDEX at
// rate p during training so the OOV embedding row receives gradient
// and attack-injected unseen IDs at inference land in a *trained*
// slot. Motivated by Monolith's low-frequency-ID filtering
// [Liu et al. 2022]; not directly citable at our >20-cite bar.
local stage = import '../../stages/supervised.jsonnet';
local paths = import '../_paths.libsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  lake_root='/fs/ess/PAS1266/graphids/dev/rf15',
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  sampler='default',
  p_unk_drop=0.1,
  trainer_overrides={}, stage_overrides={}, ckpt_path=null,
)
  stage(
    dataset=dataset, seed=seed, scale=scale,
    run_dir=paths.run_dir(lake_root, dataset, 'id_encoding', 'learned_unk', seed),
    conv_type=conv_type, loss_fn=loss_fn, sampler=sampler,
    trainer_overrides=trainer_overrides,
    stage_overrides={
      'model.init_args.id_encoder_kwargs': { p_unk_drop: p_unk_drop },
    } + stage_overrides,
    ckpt_path=ckpt_path,
  )
