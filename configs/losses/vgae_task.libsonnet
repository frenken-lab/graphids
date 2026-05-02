// VGAE task loss — class_path block. Reconstruction + CAN-ID xent +
// neighborhood xent + KL. `num_ids` is populated by VGAEModule._build at
// instantiate time (it depends on the datamodule), so it's left at default 0.

function(kl_weight=0.01, canid_weight=0.1, nbr_weight=0.05, k_neg=32)
  { loss_fn: {
      class_path: 'graphids.core.losses.VGAETaskLoss',
      init_args: {
        kl_weight: kl_weight,
        canid_weight: canid_weight,
        nbr_weight: nbr_weight,
        k_neg: k_neg,
      },
  } }
