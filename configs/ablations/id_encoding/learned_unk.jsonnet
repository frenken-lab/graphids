// Ablation arm 2/3: lookup + stochastic UNK-drop during training.
// Stage-3 design-decision arm — node_ids are remapped to UNK_INDEX at
// rate p during training so the OOV embedding row receives gradient
// and attack-injected unseen IDs at inference land in a *trained*
// slot. Motivated by Monolith's low-frequency-ID filtering
// [Liu et al. 2022]; not directly citable at our >20-cite bar.
local stage = import '../../stages/supervised.jsonnet';
local pd = (import '../../matrix/axes.json').pipeline_defaults;

function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type, loss_fn=pd.loss_fn,
  sampler='default',
  p_unk_drop=0.1,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'id_encoding', 'learned_unk', seed),
      conv_type=conv_type, loss_fn=loss_fn, sampler=sampler,
      ckpt_path=ckpt_path,
    ) + {
      model+: { init_args+: { id_encoder_kwargs: { p_unk_drop: p_unk_drop } } },
    },
    std.extVar('overrides'),
  )
