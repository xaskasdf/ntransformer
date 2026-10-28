"""Extend the already multi-patched NEW example to 64-bit host descriptor chains."""
import argparse
import hashlib
import json
import pathlib
import re

from patch_ntrpu_pcie_multi import replace_once
from patch_ntrpu_pcie_logger import patch_text


def task_body(text, name, body):
    start = text.index("task " + name + ";")
    end = text.index("endtask", start) + len("endtask")
    return text[:start] + "task " + name + ";\n    begin\n" + body + "\n    end\nendtask" + text[end:]


def patch(example):
    paths = [example / "imports" / name for name in
             ["usp_pci_exp_usrapp_tx.v", "usp_pci_exp_usrapp_rx.v", "sample_tests.vh", "usp_pci_exp_usrapp_com.v"]]
    originals = [p.read_bytes() for p in paths]
    tx, rx, tests, com = [b.decode() for b in originals]
    tx, count = re.subn(r"(localparam\s+\[15:0\]\s+DMA_BYTE_CNT\s*=\s*)16'h0300;",
                        r"\g<1>16'h0600;", tx)
    if count != 1:
        raise ValueError("Expected prior bounded multi fixture")
    tx = task_body(tx, "TSK_INIT_DATA_H2C", "        board.ntrpu_host.initialize();")
    tx = task_body(tx, "TSK_INIT_DATA_C2H",
                   '        if (!board.ntrpu_host.initialized) $fatal(1, "HOST_NOT_INITIALIZED");')
    if tx.count("payload_bytes != 768") != 2:
        raise ValueError("Unexpected payload assertions")
    tx = tx.replace("payload_bytes != 768", "payload_bytes != 1536")
    for condition in ["wbytes == 768", "c2h_bytes == 768"]:
        tx = replace_once(tx, condition, condition.replace("768", "1536"))
    start = tx.index("task TSK_TX_COMPLETION_DATA;")
    end = tx.index("endtask", start)
    completion = tx[start:end]
    completion, count = re.subn(r"input\s+\[10:0\]\s+lower_addr_;", "input [63:0] lower_addr_;", completion)
    if count != 1:
        raise ValueError("Unexpected completion address port")
    completion, count = re.subn(r"DATA_STORE\[lower_addr_\s*\+\s*([^\]]+)\]",
                                r"board.ntrpu_host.read_completion(lower_addr_, len_*4, \1)", completion)
    if count != 296 or "DATA_STORE" in completion:
        raise ValueError("Unexpected completion byte expressions: " + str(count))
    tx = tx[:start] + completion + tx[end:]
    rx, count = re.subn(r"reg\s+\[10:0\](\s+lower_addr;)", r"reg [63:0]\1", rx)
    if count != 1:
        raise ValueError("Unexpected CQ address register")
    start = rx.index("task TSK_BUILD_CPLD_PKT;")
    end = rx.index("endtask", start)
    completion = rx[start:end]
    if completion.count("cq_data[10:2]") != 5:
        raise ValueError("Unexpected CQ address extraction")
    completion = completion.replace("cq_data[10:2]", "cq_data[63:2]")
    completion = replace_once(completion, "lower_addr = 11'h0", "lower_addr = 64'h0")
    completion = replace_once(completion, "lower_addr[10:0]", "lower_addr[63:0]")
    rx = rx[:start] + completion + rx[end:]
    start = tests.index('else if(testname =="dma_test0")')
    end = tests.index('else if(testname =="dma_test1")', start)
    test = tests[start:end]
    for register, high in [("4080", 1), ("5080", 3)]:
        pattern = r"(board\.RP\.tx_usrapp\.TSK_XDMA_REG_WRITE\(16'h" + register + r",[^\n]+;)"
        test, count = re.subn(pattern, r"\1\n      board.RP.tx_usrapp.TSK_XDMA_REG_WRITE(16'h" + register[:3] +
                             "4, 32'd" + str(high) + ", 4'hF);", test)
        if count != 1:
            raise ValueError("Unexpected descriptor register setup")
    if test.count("P_READ_DATA == 32'h1") != 2 or test.count("desc_count != 1") != 2:
        raise ValueError("Unexpected descriptor-count assertions")
    test = test.replace("P_READ_DATA == 32'h1", "P_READ_DATA == 32'h2")
    test = test.replace("desc_count = 1;", "desc_count = 2;").replace("desc_count != 1", "desc_count != 2")
    for direction in ("H2C", "C2H"):
        marker = f'$display ("---***ERROR*** {direction} Descriptor count mismatch,Loop Timeout occured ---\\n");'
        test = replace_once(test, marker, f'$fatal(1, "HOST64_{direction}_DESCRIPTOR_COUNT");')
        anchor = f'      {"h2c" if direction == "H2C" else "c2h"}_status = P_READ_DATA;'
        test = replace_once(test, anchor, anchor + f'\n      $display("NTRPU_HOST64_DESCRIPTORS direction={direction} count=%0d", desc_count);')
    tests = tests[:start] + test + tests[end:]
    tx, rx, com = patch_text(tx, rx, com)
    meta = {"address_bits": 64, "payload_bytes": 1536, "descriptors_per_direction": 2,
            "bytes_per_descriptor": 768, "page_offset": 4032, "ep_axi_start": 0,
            "host_region_high_words": [1, 2, 3, 4], "completion_byte_expressions": 296,
            "logger": "Normalized bounded RQ/CC layout, 16 remainder values, 64-bit CQ write format",
            "scope": "Bounded sparse host regions; no IOMMU, P2P or physical memory qualification", "files": []}
    for path, original, text in zip(paths, originals, [tx, rx, tests, com]):
        path.write_bytes(text.encode())
        meta["files"].append({"path": str(path.relative_to(example)),
                              "before_sha256": hashlib.sha256(original).hexdigest(),
                              "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (example / "NTRPU_HOST64_PATCH.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=pathlib.Path)
    patch(parser.parse_args().example)
