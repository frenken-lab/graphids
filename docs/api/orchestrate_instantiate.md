# Orchestrate: Instantiate

Resolves every ``{class_path, init_args}`` block in a rendered config
into a live Python object. Nested blocks recurse;
``filter_kwargs(klass, init_args)`` drops kwargs the target class
doesn't accept so jsonnet stays flexible. A ``VRAMDriftCallback`` is
appended automatically when CUDA is available.

## `graphids.orchestrate.instantiate`

::: graphids.orchestrate.instantiate
