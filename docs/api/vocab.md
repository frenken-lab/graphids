# Data: Vocab

Shared-vocabulary construction for datasets with categorical node IDs
(CAN arbitration IDs, sensor names). Every split — train, val, every
test subdir — uses the same ``arb_id → index`` map so an embedding
table sized for train doesn't over-flow when a test subdir contains
attack-injected IDs. Index 0 is reserved for UNK; a SHA256 digest over
the ``(id, index)`` pairs is the cache invariant.

Stage 1 of the OOV handling plan (``~/plans/oov-embedding-handling.md``).

## `graphids.core.data.vocab`

::: graphids.core.data.vocab
