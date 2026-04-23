# Jsonnet Rendering

Thin wrapper over the ``_jsonnet`` C bindings. TLAs are JSON-serialized
via ``json.dumps`` so jsonnet receives real typed values — ints stay
ints, bools stay bools, ``None`` becomes jsonnet ``null``. The binding
is imported lazily inside
[``render``](#graphids.config.jsonnet.render) so this module stays
safe to import on login nodes without ``_jsonnet`` installed.

## `graphids.config.jsonnet`

::: graphids.config.jsonnet
