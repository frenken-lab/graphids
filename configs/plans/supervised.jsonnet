// OFAT (one-factor-at-a-time) GAT ablation plan.
//
// Returns `{ nodes: [Node, ...] }` for `graphids run` to consume.
// Each Node is one `graphids submit` call; deps reference other nodes
// by name. Test peers are explicit (cpu/long, 32 GB, 30 min).
//
// Prerequisite: a FINISHED vgae run from unsupervised.jsonnet.
// curriculum_vgae auto-derives the ckpt path from (dataset, seed);
// submit with --depends-on vgae:N if the VGAE run is still in progress.
//
// Single source of truth for the GAT ablation topology. To add a variant:
// add a lib.fit_test(...) line. To change test resources: edit _lib.libsonnet.

local lib = import '_lib.libsonnet';

function(dataset, seed)
{
  nodes:
    // gat_sampling axis — does curriculum attack-ratio ramping help?
    lib.fit_test('gat_sampling/none.jsonnet')
    + lib.fit_test('gat_sampling/curriculum_random.jsonnet')
    // gat_loss axis — which loss function handles class imbalance best?
    + lib.fit_test('gat_loss/ce.jsonnet')
    + lib.fit_test('gat_loss/weighted_ce.jsonnet')
    + lib.fit_test('gat_loss/focal.jsonnet')
    + lib.fit_test('gat_loss/tau_norm.jsonnet')
    // id_encoding axis — how should unseen CAN IDs be handled?
    + lib.fit_test('id_encoding/lookup.jsonnet')
    + lib.fit_test('id_encoding/learned_unk.jsonnet')
    + lib.fit_test('id_encoding/hash.jsonnet')
    // scaler axis — benign-only z-score vs robust (median+IQR) scaler.
    // Each arm builds its own cache (sc: prefix in cache_key).
    + lib.fit_test('scaler/z_benign.jsonnet')
    + lib.fit_test('scaler/robust_benign.jsonnet')
    // curriculum_direction axis — given attack-ratio ramping, does
    // balanced→imbalanced or imbalanced→balanced ordering matter?
    + lib.fit_test('curriculum_direction/low_to_high.jsonnet')
    + lib.fit_test('curriculum_direction/high_to_low.jsonnet')
    // curriculum_vgae: VGAE scorer, ckpt auto-derived from (dataset, seed).
    // Cross-plan dep on unsupervised.jsonnet vgae — baked into submit_command.
    + lib.fit_test('gat_sampling/curriculum_vgae.jsonnet', cross_plan_deps=['vgae']),
}
