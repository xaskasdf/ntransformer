# HBM Loader Bus-Skew Routing Experiment

## Scope

This isolated VU47P experiment starts from the intermediate BREADY checkpoint
`6f383d4e9e400c9b5215d920288a8b64385864eca48021a633b247d2a48d73aa`.
Both BREADY channels pass timing there, but the loader payload crossing has
1.190 ns actual skew against a 1.111 ns requirement: -0.079 ns slack.

The slow member is `weight_loader/active_reg[25]/Q` to
`weight_loader/address_reg[25]/D`. Its net has two loads and 1.206 ns routed
delay on the reported path. The reference member terminates at
`weight_loader/payload_reg[149]/D`.

## Native operation

`inspect_hbm_loader_skew_route.tcl` unroutes only
`weight_loader/active_reg_n_0_[25]`, reroutes that net with
`route_design -nets ... -delay`, then finalizes routing. No cell movement,
logic change, reset replication, clock repair or constraint change is requested.
The [AMD route_design reference](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/route_design)
documents this individual-net delay mode. It minimizes interconnect delay,
not the complete set of timing constraints; full-board checks remain required.

The runner uses two CPUs, nice 10 and a 1,800-second limit. Each closed batch
is archived and verified before an independent checkpoint reopen is launched.

## Evidence and acceptance

Review compares all 120,241 primitives, 143,193 canonical net routes, 475,739
hierarchical net names, both BREADY cones, 8,321 clock pins and 2,705 clock PIPs.
Each normalized canonical route is streamed to a SHA-256 digest with its byte
count, routing status and fixed flag. ECO snapshots cover before and after;
the read-only reopen supplies one independent snapshot for comparison with
the saved result. Only the specified net may change route. All prior bounded
path, CDC, reset, coverage, timing and route-completeness checks remain active.

A successful routing step must close bus-skew without degrading the direct
donor's global WNS/TNS or failure counts, and preserve BREADY and other targeted
measurements. Full candidate acceptance separately compares against the original
pre-placement design. A successful routing step is not full-board qualification.

Evidence is stored in
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-loader-skew-route/`.
`status.py` reads the private job; `advance.py` collects and reviews it, then
launches its independent reopen. `COMPARISON.json` distinguishes routing-step
acceptance from full-candidate acceptance. `FINAL_BACKUP.json` is written only
after ZIP restoration, source/preparation checks and private-process auditing.

## Diagnostic batch

The first batch, `board-loader-skew-route-inspect-20261015T123009Z`, was stopped
before routing because raw hierarchical ROUTE strings repeatedly expanded
constant-net aliases. Its partial 1,982,930,944-byte inventory and failure
archive are preserved. The replacement records canonical route digests and
all logical names; it does not omit routing scope from the audit.

Execution is in progress. Reset/control distribution remains a separate
experiment so its effects can be measured independently of this route change.
