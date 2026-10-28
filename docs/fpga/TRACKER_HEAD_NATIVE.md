# Cached-head native HBM requalification

## Purpose and boundary

Requalify the cached-head reader from [TRACKER_HEAD_CACHE.md](TRACKER_HEAD_CACHE.md)
against the native HBM simulation model. Its selected routed checkpoint is
`3b37b0ec666f6cae78679510b2880f370fcf4cc9a77f02a75b8745a8db915646`.
The standalone tracker and the embedded tracker inside the adapter must match
that checkpoint's RTL sources. Historical native results remain historical;
they do not qualify this changed RTL.

This is an isolated side experiment. It does not replace reference RTL or edit
the principal Vivado queue. The native simulation is behavioral RTL with vendor
IP; it is not a native-controller post-route simulation, integrated timing
closure, PCIe interoperability test, or board qualification.

## Reproducible inputs

`tools/prepare_ntrpu_tracker_head_native.py` clones the frozen selected source
tree, checks its reviewed result and checkpoint hashes, and preserves every
existing source. It adds a separately named bench, scripts and
`HEAD_NATIVE_PROFILE.json`. Exactly two historical RTL paths differ from the
old native profile: the standalone tracker alias and its embedded adapter copy.
All experimental and RTL source hashes are bound to the new profile.

The current mapped-region auditor is frozen under a new support filename. Its
domain wait accounting and cross-stack native AR-to-RLAST overlap check are
retained without rewriting older sealed evidence.

## Validation stages

1. **Preflight:** run the original four mapped-region scenarios with the new
   observed top and a behavioral HBM stub, using two CPUs.
2. **Compile:** check all six read/write IP configurations and the native binding
   against a reviewed native cache; explicitly recompile the changed adapter
   and elaborate `tb_ntrpu_tracker_head_native` with `NATIVE=1` on eight CPUs.
   Reusing IP compilation does not reuse the old qualified top.
3. **Traffic:** run that new snapshot in a private directory. Check initialization,
   unique preload, PC0/PC16 mapping, remapping, concurrent reads, payload,
   ownership, DMA/ACK retirement, and retained obligations after halt.

Each command has a 1,800-second bound and runs at reduced priority. Compilation
and traffic require available capacity; no existing job is interrupted.

## Cached-head observation and evidence

The new bench observes each embedded tracker without driving it. Every occupied
cycle checks cached length against the descriptor at the validation pointer.
Reservation/release events feed an independent FIFO audit, including peak
occupancy, reloads, bypasses and final conservation. Missing events, corrupted
head lengths and missing summaries must be rejected by negative controls.

Per-stage artifacts include frozen sources, manifests, reports and logs. The
compiled native runtime is copied locally with hashes, sizes and Linux mode
metadata. Each stage is added to the content-addressed incremental store,
restored to a fresh directory, and audited again before reporting it complete.

Local working evidence:
`D:/AI/artifacts/ntrpu-tracker-head-native/20261008T213956068404Z`.

## Current result

Preflight `20261008T214043449959Z-preflight1` passed in 75.290 seconds and its
221-file incremental backup passed a fresh restore audit. Both trackers reached
eight pending descriptors. Lane 0 reserved/released 32/31; lane 1 reserved/released
44/43. Each intentionally retained one descriptor after halt. The independent
auditor checked 23/35 head reloads, nine bypasses per lane, and rejected all three
corrupted head traces, alongside the existing memory/ownership/map controls.

All 14,012 public `SN`, `SG`, `SF` and `RG` rows, including timestamps, exactly
match the prior mapped preflight. This is sampled regression evidence, not an
exhaustive equivalence proof.

Native compilation `20261008T214232810839Z-compile1` passed in 500.353 seconds;
the new snapshot contains 15 runtime files totaling 511,193,674 bytes. Its
272-file incremental backup passed a fresh restore audit, including runtime
hashes, sizes and executable-mode metadata. Warning codes and counts match the
reference compilation. The first 90-second archive-transfer attempt timed out;
only the collector changed to ZIP level 1 and a 300-second transfer bound.
The compiled sources and evidence were unchanged, and the retry passed.

Native traffic `20261008T220102015280Z-traffic1` passed in 1,057.754 seconds.
The independent auditor and its fresh local restore both returned
`HEAD_NATIVE_TRAFFIC_REVIEWED`:

| Observation | Native result |
| --- | --- |
| Initialized data | 24,576 unique bytes; six writes; 48 native B responses |
| Reads | 122 AR; 70 targeting PC0 and 52 targeting PC16 |
| Verified data | 59,776 bytes: 24,576 direct and 35,200 shared |
| Ownership | 22 requests; 18 admitted; four rejected; 16 successful DMA/ACK retires |
| Halt | Two failed contexts retained; one descriptor retained per reader |
| Head checks | 1,108 / 1,377 occupied-cycle checks; peak eight descriptors per reader |
| Mapping | Two generations; concurrent native reads across different stacks |
| Corrupted evidence | All 34 memory, ownership, map and head controls rejected |

All 50,446 public trace rows, including timestamps, match the native reference
exactly. Canonical trace SHA-256:
`d88686f794b6ed7c9be3f794af64de49f468a2465597dcbeac6d1bb9d3ecad41`.
This qualifies the selected RTL for these native-controller scenarios; it does
not turn sampled regression into exhaustive equivalence. The short locality
windows reproduced 190/208/464 active cycles. Thermal CATTRIP observations
remain unknown (`X`); there is no thermal qualification.

## Sealed local evidence

The incremental store is `D:/AI/artifacts/ntrpu-content-store`. Each manifest
below was restored and re-audited locally; executable mode metadata is retained
for Linux recovery. This does not claim replay on a second Vivado installation.

- `tracker-head-native-20261008T214043449959Z-preflight1`: 221 files.
- `tracker-head-native-20261008T214232810839Z-compile1`: 272 files, including runtime.
- `tracker-head-native-20261008T220102015280Z-traffic1`: 204 files.

Native result SHA-256:
`fa951962bbf66d1753c6f855b05af669d56a720da2ca31a4232172820f1af62a`.
`OUTCOME.json` and the handoff manifest collect the receipts, source bindings,
reference comparisons and next gate. Temporary transfer archives are not kept.

## Next integration gate

The selected HBM design targets `xcvu47p-fsvh2892-2-e`. The documented
[PCIe bypass experiment](board/PCIE_BYPASS.md) targets `xcvu5p-flva2104-2-i`
and uses PIPE simulation. Those are separate qualifications: first generate and
check the PCIe configuration for the selected HBM part, then validate their
combined clocks, resets, interface constraints and implementation.

Before that physical join, an isolated bridge bench must bind the reader's
512-bit data, destination address and context to DMA submission, distinguish
local DMA completion from remote ACK, and retain ownership on link loss.
`ntrpu_peer_read_pkg` describes internal structs, not a PCIe wire ABI. Map
generation, publication and source authorization must describe the same loaded
bytes; no BAR aperture alone establishes aggregate capacity or coherence.
Retain tests for high address bits, mapped gaps, backpressure, remapping while
busy, duplicate/stale events, partial failure and coordinated cold reset.

Measure sustained bandwidth separately before increasing the buffer pool.
The three short native locality windows are functional routing checks; they
do not establish a sustained throughput target or an optimal pool depth.
