// Trainer base only — universal across every preset.
// Callbacks dict is owned by `_lib/callbacks.libsonnet`; the trainer's
// callback LIST is late-bound from `$.callbacks` at the preset apex so
// any callback added (universal trio + composer/preset extras) is
// automatically picked up by Lightning at instantiation.

{
  trainer: {
    accelerator: 'auto',
    devices: 'auto',
    precision: '16-mixed',
    max_epochs: 300,
    gradient_clip_val: 1.0,
    callbacks: [$.callbacks[k] for k in std.objectFields($.callbacks)],
  },
}
