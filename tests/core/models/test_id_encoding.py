"""Id-encoder contract tests: LookupIdEncoder + HashIdEncoder.

Framework-level tests (shape, dtype) are nn.Embedding's job and
deliberately omitted per tests/.../test-writing.md.
"""

from __future__ import annotations

import torch

from graphids.core.models.id_encoding import (
    UNK_INDEX,
    HashIdEncoder,
    LookupIdEncoder,
    build_id_encoder,
    encoding_plan,
    hash_encoding,
    lookup_encoding,
)


def test_p_unk_drop_one_routes_all_ids_to_unk_row():
    # CONTRACT: at p=1.0 in training, every id becomes UNK_INDEX, so
    # the whole output equals the UNK row broadcast over the batch.
    enc = LookupIdEncoder(num_ids=10, embedding_dim=4, p_unk_drop=1.0)
    enc.train()
    out = enc(torch.arange(5))
    unk_row = enc.embedding.weight[UNK_INDEX]
    assert torch.allclose(out, unk_row.expand_as(out))


def test_eval_mode_ignores_p_unk_drop():
    # CONTRACT: like nn.Dropout, UNK-drop is train-only. Eval must
    # return the direct embedding lookup regardless of p.
    enc = LookupIdEncoder(num_ids=10, embedding_dim=4, p_unk_drop=1.0)
    enc.eval()
    node_id = torch.tensor([1, 2, 3])
    assert torch.allclose(enc(node_id), enc.embedding(node_id))


def test_unk_row_receives_gradient_under_unk_drop():
    # REGRESSION: the whole point of Stage 3 is that the UNK row
    # gets trained. Assert its gradient is non-zero after one backward
    # with p_unk_drop>0. Without UNK-drop the UNK row would only see
    # gradient when an actual OOV id appears at inference — too rare
    # to learn a useful slot.
    enc = LookupIdEncoder(num_ids=10, embedding_dim=4, p_unk_drop=1.0)
    enc.train()
    out = enc(torch.arange(5))
    out.sum().backward()
    grad = enc.embedding.weight.grad
    assert grad is not None
    assert grad[UNK_INDEX].abs().sum() > 0


def test_hash_same_id_deterministic_across_forwards():
    # CONTRACT: the core hashing invariant. Same id must produce the
    # same embedding across calls — otherwise the encoder has no
    # meaning to learn.
    enc = HashIdEncoder(num_buckets=64, embedding_dim=8, k=2, seed=42)
    enc.eval()
    node_id = torch.tensor([3, 7, 42, 999])
    assert torch.allclose(enc(node_id), enc(node_id))


def test_hash_seed_changes_bucket_assignment():
    # CONTRACT: seed is respected. Two encoders with same weights but
    # different seeds must probe different buckets → different outputs.
    # Guards against a future refactor where seed is accepted but
    # silently ignored by the hash function.
    enc_a = HashIdEncoder(num_buckets=64, embedding_dim=8, k=2, seed=42)
    enc_b = HashIdEncoder(num_buckets=64, embedding_dim=8, k=2, seed=99)
    # Copy only the embedding table so the only remaining difference
    # is the hash offsets driven by seed.
    enc_b.embedding.load_state_dict(enc_a.embedding.state_dict())
    enc_a.eval()
    enc_b.eval()
    node_id = torch.tensor([3, 7, 42])
    assert not torch.allclose(enc_a(node_id), enc_b(node_id))


def test_hash_unknown_id_deterministic():
    # REGRESSION: the Stage-2 research claim. An id never seen at
    # training time (here: id = 10_000, vastly outside the vocab that
    # seeded the table) must still map to a fixed bucket pattern and
    # produce a reproducible embedding across calls. This is the
    # mechanism that replaces the UNK slot — no special case needed.
    enc = HashIdEncoder(num_buckets=64, embedding_dim=8, k=2, seed=42)
    enc.eval()
    unseen = torch.tensor([10_000])
    assert torch.allclose(enc(unseen), enc(unseen))


def test_hash_k_probes_contribute():
    # REGRESSION: guards the k-probe design. k=1 collapses to a
    # single-hash embedding; k=2 must produce a different output
    # because a second probe is added. If a refactor silently breaks
    # the sum-over-probes pathway, k would stop mattering.
    torch.manual_seed(0)
    enc_k1 = HashIdEncoder(num_buckets=64, embedding_dim=8, k=1, seed=42)
    torch.manual_seed(0)  # same table init
    enc_k2 = HashIdEncoder(num_buckets=64, embedding_dim=8, k=2, seed=42)
    enc_k1.eval()
    enc_k2.eval()
    # Share the embedding table so only the probe count differs.
    enc_k2.embedding.load_state_dict(enc_k1.embedding.state_dict())
    node_id = torch.tensor([3, 7, 42])
    assert not torch.allclose(enc_k1(node_id), enc_k2(node_id))


def test_hash_offsets_survive_checkpoint_roundtrip():
    # CONTRACT: _hash_offsets is registered as a buffer (not a plain
    # attribute) so state_dict save/load preserves it. Without this,
    # a reloaded ckpt constructed with a different seed would produce
    # the wrong hash functions and silently corrupt every embedding
    # lookup. This is the architectural promise behind ``register_buffer``.
    enc_orig = HashIdEncoder(num_buckets=64, embedding_dim=8, k=2, seed=42)
    enc_orig.eval()
    state = enc_orig.state_dict()

    enc_new = HashIdEncoder(num_buckets=64, embedding_dim=8, k=2, seed=99)
    enc_new.load_state_dict(state)
    enc_new.eval()

    node_id = torch.tensor([3, 7, 42])
    assert torch.allclose(enc_orig(node_id), enc_new(node_id))


def test_id_encoding_config_helpers_round_trip():
    cfg = lookup_encoding(embedding_dim=6, p_unk_drop=0.25)
    plan = encoding_plan(cfg)
    enc = build_id_encoder(cfg, num_ids=10)
    assert plan.kind == "lookup"
    assert enc.out_dim == 6

    hcfg = hash_encoding(embedding_dim=7, k=3, seed=11, num_buckets=32)
    hplan = encoding_plan(hcfg)
    henc = build_id_encoder(hcfg, num_ids=10)
    assert hplan.kind == "hash"
    assert henc.out_dim == 7
