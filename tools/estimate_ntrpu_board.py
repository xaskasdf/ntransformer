"""Estimate board memory capacity and streaming bounds; no hardware validation.

Outputs JSON to stdout. GiB = 2**30 bytes; GB/s = 10**9 bytes/second.
Assumes balanced channel traffic and ideal fabric utilization. DDR efficiency
is a user assumption, not a measurement. Compute and PCIe are not modeled.
"""

import argparse
import json
import math


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=positive_int, default=4)
    parser.add_argument("--dimms-per-channel", type=positive_int, default=2)
    parser.add_argument("--dimm-gib", type=positive_int, default=64)
    parser.add_argument("--data-rate-mts", type=positive_float, default=2400)
    parser.add_argument("--payload-bits", type=positive_int, default=64)
    parser.add_argument("--ddr-efficiency", type=positive_float, default=0.7)
    parser.add_argument("--fabric-ports", type=positive_int, default=4)
    parser.add_argument("--fabric-bits", type=positive_int, default=512)
    parser.add_argument("--fabric-mhz", type=positive_float, default=200)
    parser.add_argument("--weight-gb", type=positive_float, default=35)
    args = parser.parse_args()
    if args.ddr_efficiency > 1:
        parser.error("--ddr-efficiency must be in (0, 1]")
    if args.payload_bits % 8 or args.fabric_bits % 8:
        parser.error("bus widths must be multiples of 8 bits")

    channel_bytes = args.dimms_per_channel * args.dimm_gib * 2**30
    capacity_bytes = args.channels * channel_bytes
    ddr_gbps = args.channels * (args.payload_bits / 8) * args.data_rate_mts / 1000
    assumed_ddr_gbps = ddr_gbps * args.ddr_efficiency
    fabric_gbps = args.fabric_ports * (args.fabric_bits / 8) * args.fabric_mhz / 1000
    limiting_gbps = min(assumed_ddr_gbps, fabric_gbps)
    weight_bytes = args.weight_gb * 10**9
    if not all(math.isfinite(value) and value > 0 for value in
               (ddr_gbps, assumed_ddr_gbps, fabric_gbps, limiting_gbps, weight_bytes)):
        parser.error("inputs produce non-finite or zero bandwidth/size")

    result = {
        "status": "analytical_scenario_not_hardware_validation",
        "inputs": vars(args),
        "capacity": {
            "dimm_count": args.channels * args.dimms_per_channel,
            "bytes_before_runtime_reservations": capacity_bytes,
            "gib_before_runtime_reservations": capacity_bytes // 2**30,
            "channel_address_bits": (channel_bytes - 1).bit_length(),
            "board_address_bits": (capacity_bytes - 1).bit_length(),
            "weights_fit_before_other_allocations": weight_bytes <= capacity_bytes,
        },
        "bandwidth_gb_per_second": {
            "ddr_payload_theoretical": ddr_gbps,
            "ddr_with_assumed_efficiency": assumed_ddr_gbps,
            "fabric_theoretical": fabric_gbps,
            "streaming_ceiling_under_assumptions": limiting_gbps,
        },
        "weight_pass": {
            "seconds_lower_bound_under_assumptions": args.weight_gb / limiting_gbps,
            "passes_per_second_ceiling_under_assumptions": limiting_gbps / args.weight_gb,
        },
        "limitations": [
            "Does not validate DIMM, FPGA, pinout, IP support or achievable clocks.",
            "ECC bits are excluded from payload; reserve memory separately for KV and buffers.",
            "No compute, attention, KV, refresh model, contention or PCIe timing simulation.",
            "Weight passes are not measured inference tokens per second.",
        ],
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
