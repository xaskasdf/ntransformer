# Llama 3.2 normalization/Q8: actual C/RTL results

Run `20261013T155333Z-V7-llama32-norm-q8-hls-r1` completed on
2026-10-13 at 16:53:16 UTC. Vitis 2026.1 targeted `xcvu5p-flva2104-2-i`
with a 5 ns clock. Independent review passed, and the terminal archive
was verified locally on D: and on NFS before the attention queue started.

## Numerical and transaction coverage

All 61 transactions passed: 49 successful cases and 12 rejected inputs.
The successful output contains 2,860 ABI1 records, 22,880 64-bit words,
or 183,040 bytes. These match the independent ordered rational/Decimal
reference and the actual C/RTL output frames exactly. Fixtures include
four actual embedding/layer-0 normalization paths, plus numerical and
shape edge cases.

Each rejected transaction preserves all 512 output words at their
sentinel value in both C and RTL frames. The numerical reviewer rejects
10 corrupted transcripts; the transaction reviewer rejects another 10
controls, including matching wrong C/RTL words, invalid scales, reserved
bytes, missing frames and invalid latency records.

## Measured simulation and estimates

| Measurement | Result |
| --- | ---: |
| Transaction latency, minimum / average / maximum | 2 / 122,856 / 167,188 cycles |
| Transaction interval, minimum / average / maximum | 3 / 124,870 / 167,189 cycles |
| Co-simulation execution | 7,494,306 cycles |
| Maximum latency at the simulated 5 ns clock | 835.94 microseconds |
| HLS BRAM18K / DSP / FF / LUT | 27 / 11 / 3,678 / 9,210 |
| HLS estimated period | 3.647 ns |

The bank mixes successful and rejected workloads. These timings describe
local `ap_memory` transactions; they exclude AXI, DDR, attention, host
communication and complete inference. Resource and clock estimates are
pre-route and do not establish physical timing closure.

## Runtime and preserved evidence

Elapsed tool time was 3,579.49 seconds. The bounded
[runtime extension](board/NORM_Q8_RUNTIME_EXTENSION_R1.md) preserved the
original Vitis/XSIM processes. Its guardian resumed the waiting wrapper
after the tool completed, and the wrapper recorded exit code 0.
The original 2,400-second invocation and explicit 5,400-second effective
limit both remain in the archive; no restart or source change occurred.

- Terminal archive SHA-256:
  `411d90a513f6d68acb7735e617cd0451a5fc531fd38529b5fe585ae1b1433be4`.
- Independent numerical/C-RTL review:
  `64d6610bdafd2fb6215036268708a98a0f59923af40408bbc49fa14a4bd1ef37`.
- Independent terminal-extension review:
  `d353133cea0b5f715a8d70634e89199a11eaa9b7f06d0b75e412339e957aeb61`.

Local coordinator and review records are under
`build/cloud-setup/20261002-board/`. The next dependency,
`20261013T165530Z-V7-llama32-attention-hls-r1`, started at 16:55:30 UTC.
