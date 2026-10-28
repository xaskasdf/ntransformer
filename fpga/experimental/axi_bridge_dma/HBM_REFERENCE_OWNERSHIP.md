# VU47P HBM reference-clock ownership

This isolated experiment follows [HBM_BOARD_STARTUP.md](HBM_BOARD_STARTUP.md).
It addresses the two HBM reference-buffer cascades and the internal primary
clock redefinitions found in the earlier placed composition. It preserves
that checkpoint and the board frontend's RTL.

## Integration change

Set `CONFIG.USER_DIS_REF_CLK_BUFG` to `true` before generating the native HBM
IP. Vivado 2026.1, HBM IP revision 20, accepts the property and generates:

- No native HBM reference IBUF or BUFG instances, for either stack.
- Direct reference connections into the HBM implementation.
- No active commands in `hbm_ip.xdc`; the two internal `create_clock`
  commands disappear through IP generation, without editing generated files.

The board continues to own each path:

```text
hbm_refN_clk_p/n -> board_inputs/IBUFDS -> hbm_referenceN/BUFGCE -> HBM
```

The primary clock belongs on `hbm_refN_clk_p`, with a 10 ns period. The
reserved package pairs remain BK33/BL33 and BH32/BJ32; this experiment does
not apply electrical standards or release a pinout. AMD requires an HBM
reference driven through BUFG or MMCM/BUFG, with its source GCIO in the HBM
SLR and source jitter below 3 ps RMS
([PG276 Clocking](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking)).

## Evidence

Artifacts belong to
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-reference-ownership`.
Generation `reference-owner-generate-20261013T133721Z` completed in 98.05 s.
Its archive was restored locally and all 272 evidence files verified.
Comparison with the frozen seven-IP generation checked 2,943 properties:
only `USER_DIS_REF_CLK_BUFG` changed, from `FALSE` to `TRUE`.

The board top remains SHA-256
`f73a5c8ff85e6ba70325d0a1c123ee649c030d4fdc50024e77240311bd679462`.
Startup, PCIe reset, transaction, weight-loading and availability RTL are
unchanged. Older binding JSON files copied with the candidate describe their
original donors; they are not qualification of the new IP configuration.

Native synthesis `reference-owner-synth-20261013T134139Z` produced checkpoint
SHA-256 `915921ed14373844cdb1cbc994e42e68bddb9760775ddf323e21431e33ae124b`.
Vivado's synthesis summary has zero errors, zero critical warnings and 80,971
warnings; emitted examples include unused ports and constant-folded resources.
This warning count is not a clean-design sign-off.

The first audit incorrectly compared the IBUFDS wrapper output with a leaf
pin. Vivado expands that primitive into DIFFINBUF/IBUFCTRL. The 906.67 s job
retains its `FAILED` status and all 273 evidence files. Diagnostic inspection
`reference-owner-inspect-20261013T140038Z` exposed the actual connection;
the correction is in the verifier, with no RTL or netlist modification.

Final inspection `reference-owner-inspect-20261013T140415Z` reopened the
identical checkpoint and passed in 186.97 s. It proves both pad polarities,
the exact `IBUFCTRL_INST/O` driver inside each IBUFDS, one board BUFGCE per
reference, and direct loads at the respective `HBM_REF_CLK/REF_CLK` primitive.
Each leaf sees its sole 10 ns primary clock originating at the positive pad.
The design has 75 external bits, all four native blocks and 75 timing clocks.
Before/after active XDC commands are identical; there are no waivers.

| Methodology finding | Count in reopened synthesis |
| --- | --- |
| TIMING-2 / TIMING-4 / CLKC-5 | 0 / 0 / 0 |
| AVAL-324, missing PCIe reference-buffer LOC | 1 critical |
| TIMING-6 / TIMING-7, clock relations needing constraints | 10 / 10 critical |
| LUTAR-1 / TIMING-9 / TIMING-18 | 14 / 1 / 2 warnings |

OOC methodology explicitly omits some connectivity checks. Repeat the clock
review after placement; this result does not establish electrical or routed
timing qualification. The clock inventory also logs benign MASTER_CLOCK
queries on primary clocks; acceptance uses their source, period and propagated
clock identities, not that inapplicable property.

All four stages are locally restored. The final audit verified 578 remote
evidence files, the 236-file original generation and 63 local donor files.
None of the eight owned process groups remain live or zombie. The final bundle
includes the original failed verifier, its diagnosis, the corrected audit and
all three prepared revisions. `FINAL_BACKUP.json` records full restoration,
repeated independent reviews and exact candidate reconstruction.

## Reproduction and remaining scope

```text
python -B prepare_hbm_reference_ownership.py FROZEN_FRONTEND NEW_CANDIDATE --synth-audit
python -B NEW_CANDIDATE/run_hbm_native_join.py PRIVATE_GENERATION --prepared NEW_CANDIDATE --settings VIVADO_SETTINGS --stage generate
python -B NEW_CANDIDATE/run_hbm_native_join.py PRIVATE_SYNTH --prepared NEW_CANDIDATE --settings VIVADO_SETTINGS --stage synth --generation PRIVATE_GENERATION/output
python -B review_hbm_reference_ownership.py generation RESTORED_GENERATION --baseline FROZEN_GENERATION
python -B review_hbm_reference_ownership.py synthesis RESTORED_SYNTH
```

For this archived run, append `--inspection RESTORED_FINAL_INSPECTION` to the
synthesis review. This binds the corrected read-only audit to the exact saved
checkpoint while preserving the original stage failure. The final candidate
is `prepared-r3`; r1 generated the IP and r2 ran synthesis with the earlier
verifier. Only the verifier differs between r2 and r3.

Use the runner in `NEW_CANDIDATE`: it retains the two-CPU, reduced-priority
policy and allows 1,800 seconds per tool stage. Every stage uses a fresh
directory, copies its donor, and archives its sources, reports and results.
The final checkpoint-only inspection has a shorter 600 s limit.

This synthesis does not inherit the earlier checkpoint's custom CDC/data
bounds or placement. They must be rebound and audited against this netlist,
including the new startup synchronizers, before placement/routing. Board
power sequencing, electrical clock choices, supervisor pin, routed timing,
traffic in native HBM, and hardware operation remain unqualified.
