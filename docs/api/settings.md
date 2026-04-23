# Settings

Every ``GRAPHIDS_*`` environment variable is typed on
``GraphIDSSettings`` and read once via ``get_settings()``. Adding a new
env var means adding a field here — no scattered ``os.environ.get()``.

## `graphids.config.settings`

::: graphids.config.settings
