# Python configparser: Full Feature Report

## What It Is

`configparser` (stdlib since Python 2) reads/writes INI-style config files. It models config as
sections containing key-value string pairs, with a special `DEFAULT` section whose values are
inherited by all other sections. Keys are case-insensitive (lowercased via `optionxform`); section
names are case-sensitive. All values are stored as strings internally.

## INI File Structure

```ini
[DEFAULT]
timeout = 30

[database]
host = localhost
port = 5432          # inherits timeout = 30 from DEFAULT

[app]
debug = yes
workers = 4
multiline = line one
    line two         # continuation via indentation
```

Comments: `#` or `;` on their own line (inline comments disabled by default).
Delimiters: `=` or `:`. Multiline values: indent continuation lines deeper than the key.

## Three Parser Classes

| Class | Interpolation | Notes |
|-------|--------------|-------|
| `ConfigParser` | `BasicInterpolation` (default) | Recommended. All values are strings. |
| `RawConfigParser` | `None` (disabled) | Legacy. Allows non-string `set()` (unsupported). Use `ConfigParser(interpolation=None)` instead. |
| `SafeConfigParser` | Deprecated alias for `ConfigParser` since 3.2; removed in 3.12. | Do not use. |

## Constructor Parameters

```python
ConfigParser(defaults=None, dict_type=dict, allow_no_value=False, *,
             delimiters=('=', ':'), comment_prefixes=('#', ';'),
             inline_comment_prefixes=None, strict=True,
             empty_lines_in_values=True, default_section='DEFAULT',
             interpolation=BasicInterpolation(), converters={},
             allow_unnamed_section=False)
```

Key knobs: `strict=True` rejects duplicate sections/options on read. `allow_no_value=True` permits
keys without `=` (stored as `None`). `allow_unnamed_section=True` (3.13+) allows a sectionless
preamble accessed via `UNNAMED_SECTION`.

## Reading and Writing

- **`read(filenames)`** -- silently skips missing files; returns list of files actually parsed.
  Later files override earlier ones (last-wins priority).
- **`read_file(f)`** / **`read_string(s)`** / **`read_dict(d)`** -- from file object, string, or dict.
- **`write(f, space_around_delimiters=True)`** -- serializes to file. Comments are NOT preserved.

## Type Conversion

All values are strings. Explicit conversion via convenience getters:

- `getint(section, option)`, `getfloat(section, option)`
- `getboolean(section, option)` -- recognizes `yes/no`, `on/off`, `true/false`, `1/0` (case-insensitive)
- All accept `fallback=` and `raw=True` kwargs.

### Custom Converters

```python
import decimal
parser = ConfigParser(converters={'decimal': decimal.Decimal})
parser.getdecimal('section', 'key', fallback=0)        # parser-level
parser['section'].getdecimal('key', 0)                  # section proxy
```

Each entry in `converters` auto-generates `get<name>()` on both the parser and every section proxy.

## Fallback Values (3-tier lookup)

1. Option in the requested section
2. Option in `DEFAULT` section
3. `fallback=` kwarg (only if option is missing from both tiers above; DEFAULT wins over fallback)

## Interpolation

**BasicInterpolation** (default): `%(key)s` references within the same section + DEFAULT. Escape `%` with `%%`.

**ExtendedInterpolation**: `${key}` (same section) or `${section:key}` (cross-section). Escape `$` with `$$`. Enables cross-section references -- closest thing to Jsonnet's import semantics, but flat.

**Disable**: `interpolation=None` or use `raw=True` on individual `get()` calls.

Interpolation is resolved on demand, not at parse time. `MAX_INTERPOLATION_DEPTH` prevents cycles.

## Mapping Protocol

ConfigParser supports dict-like access: `config['section']['key']`, `in` operator, iteration.
Key differences from real dicts: keys are case-insensitive, DEFAULT values bleed into every section,
`clear()` on a section does not remove inherited defaults, DEFAULTSECT cannot be deleted.

## Exceptions

All inherit from `configparser.Error`: `NoSectionError`, `NoOptionError`, `DuplicateSectionError`,
`DuplicateOptionError`, `InterpolationDepthError`, `InterpolationMissingOptionError`,
`InterpolationSyntaxError`, `MissingSectionHeaderError`, `ParsingError`, `InvalidWriteError` (3.14+).

## Strengths

- Zero dependencies (stdlib), universally available
- Simple API for flat key-value configs
- Built-in boolean parsing, fallback chains, interpolation
- Custom converters are ergonomic
- Good for end-user-facing config (INI is widely understood)

## Limitations

- **No nested structures.** Sections are one level deep; no dicts-of-dicts, no lists, no arrays.
- **No schema validation.** No way to declare required keys, types, or constraints.
- **No type safety.** Everything is a string; conversion is manual per-access.
- **Comments lost on write.** Round-trip editing destroys comments.
- **No composition/imports.** Cannot split config across files with merge semantics.
- **Case-insensitive keys only** (by default; overridable but non-standard).
- **No environment variable expansion** without custom interpolation subclass.

## Comparison with Alternatives

| Feature | configparser | TOML | YAML | Jsonnet | Pydantic Settings |
|---------|-------------|------|------|---------|-------------------|
| Nested structures | No | Yes | Yes | Yes | Yes |
| Type safety | No (strings) | Native types | Native types | JSON types | Full validation |
| Schema validation | No | No | No | No | Yes (models) |
| Composition/imports | No | No | YAML anchors | `import` + deep merge | Layered sources |
| Cross-references | Interpolation only | No | No | Full language | Computed fields |
| Stdlib | Yes | Yes (3.11+) | No | No | No |
| Human-editable | Very easy | Easy | Easy | Moderate | N/A (code) |

## Verdict for GraphIDS

**configparser is irrelevant to this stack.** GraphIDS needs deep nesting (model/trainer/callback
hierarchies), typed composition (Jsonnet `+:` deep-merge and imports), schema validation (Pydantic
`ValidatedConfig`), and list/array values (auxiliaries, callbacks, param groups). configparser's flat
section/key string model cannot represent any of these. Its `ExtendedInterpolation` is the closest
feature to cross-file references, but it only handles string substitution within a single parsed
config. configparser fits simple end-user-facing settings (CLI defaults, deployment toggles), not
ML experiment config. No action needed.
