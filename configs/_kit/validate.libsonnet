// Render-time validators for the spec apex contract.
// `v.spec(rendered)` is the apex wrapper — every spec's archetype composer
// wraps the rendered output with `v.spec(...)`, so contract violations
// surface at jsonnet eval, not at submit time.

local has = std.objectHas;

local _check_meta(m) =
  assert std.isObject(m) : '_meta must be object';
  assert has(m, 'group') && std.isString(m.group) : '_meta.group missing or not string';
  assert has(m, 'variant') && std.isString(m.variant) : '_meta.variant missing or not string';
  assert has(m, 'dataset') && std.isString(m.dataset) : '_meta.dataset missing or not string';
  assert has(m, 'seed') && std.isNumber(m.seed) : '_meta.seed missing or not number';
  assert has(m, 'model_type') && std.isString(m.model_type) : '_meta.model_type missing';
  assert has(m, 'scale') && std.isString(m.scale) : '_meta.scale missing';
  m;

// Preset-side _resources carries archetype-fixed `mode` only. `length` is a
// row-level decision (smoke vs production), set by the plan via row.fit().
local _check_resources(r) =
  assert std.isObject(r) : '_resources must be object';
  assert has(r, 'mode') && std.member(['gpu', 'cpu'], r.mode) : "_resources.mode must be 'gpu'|'cpu'";
  r;

{
  spec(rendered)::
    assert has(rendered, 'model') : 'spec missing model block';
    assert has(rendered, 'data') : 'spec missing data block';
    assert has(rendered, 'trainer') : 'spec missing trainer block';
    assert has(rendered, 'callbacks') : 'spec missing callbacks block';
    assert has(rendered, '_meta') : 'spec missing _meta';
    assert has(rendered, '_resources') : 'spec missing _resources';
    assert has(rendered, '_upstreams') : 'spec missing _upstreams (use [] if none)';
    rendered + {
      _meta: _check_meta(rendered._meta),
      _resources: _check_resources(rendered._resources),
    },
}
