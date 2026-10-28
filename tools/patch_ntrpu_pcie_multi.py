"""Patch a NEW AMD example copy for bounded multi-packet DMA validation.

No IP RTL changes. Host-memory layout and the 11-bit completion-task address
limit remain explicit. This is not a general-purpose root-complex memory model.
"""
import argparse
import hashlib
import json
import pathlib
import re


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence: {old!r}")
    return text.replace(old, new)


def patch(example):
    paths = [example / "imports" / name for name in
             ["usp_pci_exp_usrapp_tx.v", "sample_tests.vh"]]
    before = [p.read_bytes() for p in paths]
    tx, tests = [b.decode() for b in before]
    tx, n = re.subn(r"(?m)^(\s*localparam\s+\[15:0\]\s+DMA_BYTE_CNT\s*=\s*)16'h0010;",
                    r"\g<1>16'h0300;", tx)
    if n != 1:
        raise ValueError("Expected original 16-byte DMA parameter")
    start = tx.index("task TSK_INIT_DATA_H2C;")
    end = tx.index("endtask", start)
    init = replace_once(tx[start:end], "DATA_STORE[1024+k] = k;",
                        "DATA_STORE[1024+k] = (k ^ (k >> 8) ^ 'ha5);")
    tx = tx[:start] + init + tx[end:]
    start = tx.index("task TSK_TX_COMPLETION_DATA;")
    end = tx.index("endtask", start)
    completion = tx[start:end]
    # First beat already uses DATA_STORE[lower_addr_ + ...]. Keep every later
    # beat and the vendor packet log on that same memory and address base.
    completion, n = re.subn(r"DATA_STORE_2\[ram_ptr \+", "DATA_STORE[lower_addr_ +", completion)
    if n != 192:
        raise ValueError(f"Unexpected continuation-byte expressions: {n}")
    tx = tx[:start] + completion + tx[end:]
    for direction, condition in [("H2C", "board.ntrpu_monitor.wbytes == 768"),
                                 ("C2H", "board.ntrpu_monitor.cq.c2h_bytes == 768")]:
        start = tx.index(f"task COMPARE_DATA_{direction};")
        end = tx.index("endtask", start) + len("endtask")
        tx = tx[:start] + f"""task COMPARE_DATA_{direction};
    input [31:0] payload_bytes;
    begin
        if (payload_bytes != 768) $fatal(1, "Unexpected multi DMA length");
        wait ({condition});
        $display("NTRPU_MULTI_{direction}_BYTES_OBSERVED");
    end
endtask""" + tx[end:]
    start = tests.index('else if(testname =="dma_test0")')
    end = tests.index('else if(testname =="dma_test1")', start)
    test = replace_once(tests[start:end], "$finish;",
                        "board.ntrpu_monitor.verify_complete();\n    $finish;")
    test = replace_once(test, "board.RP.tx_usrapp.TSK_INIT_DATA_H2C;", """// Configure after enumeration: vendor tag setup rewrites Device Control.
      TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG, DEV_CTRL_REG_ADDR, 4'hF);
      DEFAULT_TAG = DEFAULT_TAG + 1;
      TSK_WAIT_FOR_READ_DATA;
      TSK_TX_TYPE0_CONFIGURATION_WRITE(DEFAULT_TAG, DEV_CTRL_REG_ADDR,
          (P_READ_DATA & 32'h00008f1f) | 32'h00001020, 4'h3);
      DEFAULT_TAG = DEFAULT_TAG + 1;
      TSK_TX_CLK_EAT(100);
      TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG, DEV_CTRL_REG_ADDR, 4'hF);
      DEFAULT_TAG = DEFAULT_TAG + 1;
      TSK_WAIT_FOR_READ_DATA;
      $display("NTRPU_MULTI_DEVCTL_READBACK value=%h", P_READ_DATA);
      if (P_READ_DATA[7:5] !== 3'd1 || P_READ_DATA[14:12] !== 3'd1)
          $fatal(1, "NTRPU_MULTI_CONFIG_MISMATCH");
      $display("NTRPU_MULTI_CONFIG ep_mps=256 ep_mrrs=256 devctl=%h", P_READ_DATA);
      board.RP.tx_usrapp.TSK_INIT_DATA_H2C;""")
    tests = tests[:start] + test + tests[end:]
    metadata = {"payload_bytes": 768, "host_h2c": [1024, 1792],
                "host_c2h": [2048, 2816], "descriptor_addresses": [256, 768],
                "completion_address_bits": 11, "ep_mps": 256, "ep_mrrs": 256,
                "devctl_programming": "read-modify-write after vendor enumeration, low two bytes only",
                "pattern": "byte(i ^ (i >> 8) ^ 0xa5)",
                "continuation_byte_expressions_repaired": n, "files": []}
    for path, original, new in zip(paths, before, [tx, tests]):
        data = new.encode()
        path.write_bytes(data)
        metadata["files"].append({"path": str(path.relative_to(example)),
                                   "before_sha256": hashlib.sha256(original).hexdigest(),
                                   "after_sha256": hashlib.sha256(data).hexdigest()})
    (example / "ntrpu_multi_patch.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=pathlib.Path)
    patch(parser.parse_args().example)
