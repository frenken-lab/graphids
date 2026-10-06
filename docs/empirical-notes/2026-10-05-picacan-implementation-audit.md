# PicaCAN implementation audit

The prior experimental implementation was incomplete. This audit compares its code to the locally available paper and the paired bus-0 fitting captures. It is not a PicaCAN retrieval result. The fixed five-node ICE graph remains a separate baseline; the combined chassis graph cannot substitute for it.

## Defects corrected in this pass

- Graph matching ranked outgoing delays against every TECG node. Section V-C of the paper ranks the outgoing edges in the **selected subgraph**. The matcher now ranks within each five-node candidate and enumerates topology-compatible candidates without iterating every five-permutation.
- The bit-flip candidate pool previously retained obvious status and modular counter-like fields. The fit now applies a fit-only filter with an explicit four-state threshold and a modular-increment check. This is an operational rule; checksum exclusion and the paper's full multi-byte extraction remain incomplete.
- Assessment previously reused the fitting match status and checked only activity. It now recomputes edges and matching from frozen fields and semantic groups on each assessment route, with route-local observation and change gates. It does not reselect fields or pivots.
- RatioDSTE previously rebuilt symbols and Python histograms for every lag. It now symbolizes once per pair and uses vectorized categorical entropy; a 2,000-sample, 30-delay pilot fell from about 1.31 s to 0.047 s. This is a speed correction, not a validation of the full method.
- Cohort preparation no longer labels a catalog platform as verified ICE merely from its platform name. The manifest records `unverified`; fitting requires explicit verified ICE metadata.

## Capture-level obstacle

The pinned OpenDBC and local fitting CSVs show the following **message presence**, not validated active physical fields:

| Platform | Present in two fit captures | Relevant absence |
| --- | --- | --- |
| Civic | `0x17C` pedal gas/RPM; `0x158` RPM; `0x1D0` four wheel speeds; `0x309` car speed; `0x1E7` brake pressure | No separately named throttle-position signal in the observed DBC messages |
| Sonata | `0x162` cluster RPM; `0x52A` vehicle speed; `0x220` yaw and acceleration | DBC accelerator-pedal IDs `0x091` and `0x371` do not occur in either fit capture |
| Forester | `0x040` RPM and throttle pedal/combination; `0x139` brake pedal | No demonstrated separate throttle-plate observation from these messages |

PicaCAN's ICE SECG distinguishes accelerator pedal and throttle position as different nodes. A five-node result therefore needs either an independently discovered unlabelled field, another bus/capture, or a different vehicle. The paired DBC cannot certify a field that is absent from the captured bus. Do not collapse pedal and throttle labels to force a match.

## Still required before a paper-comparable claim

The baseline still lacks verified checksum removal and complete Motorola/Intel multi-byte extraction; a robust earliest-pivot algorithm and the full synthetic controls; the long, frozen contiguous cohort and powertrain metadata; physical-factor and post-fit DBC-oracle ablations; and a completed fit/development/sealed run. A full run is expensive: the Civic two-route pilot produced 90 bit-flip candidates among the 16 most frequent IDs before status filtering. Pairwise delayed Pearson and 300-delay RatioDSTE must be measured and scheduled on the intended cohort. Until then, five-role retrieval is **unmeasured**, rather than a failed or successful replication.
