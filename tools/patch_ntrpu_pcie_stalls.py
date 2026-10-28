"""Insert simulation-only AXI W/R stalls into a NEW generated example copy."""
import argparse
import hashlib
import json
import pathlib
import re


def patch(root, all_channels=False):
    path = root / "imports/xilinx_dma_pcie_ep.sv"
    before = path.read_bytes()
    text = before.decode()
    if "ntrpu_w_gate" in text:
        raise ValueError("Example already contains simulation gates")
    anchor = "  // XDMA target application"
    if text.count(anchor) != 1:
        raise ValueError("Unexpected application declaration anchor")
    text = text.replace(anchor,
                        "  wire ntrpu_mem_wvalid, ntrpu_mem_wready, ntrpu_mem_rvalid, ntrpu_mem_rready;\n" + anchor)
    # Rewire only the public example application ports, leaving the DMA IP untouched.
    for name in ("wvalid", "wready", "rvalid", "rready"):
        pattern = r"(\.s_axi_" + name + r"\s*\()m_axi_" + name + r"(\))"
        text, count = re.subn(pattern, r"\g<1>ntrpu_mem_" + name + r"\2", text)
        if count != 1:
            raise ValueError("Unexpected application port: " + name)
    insertion = '''
  // NTRPU SIMULATION ONLY. This copied project must not be implemented.
  ntrpu_sim_ready_gate #(.WIDTH(577), .PERIOD(11), .BLOCKED(6), .CHANNEL("W")) ntrpu_w_gate (
    .clk(user_clk), .resetn(user_resetn),
    .s_valid(m_axi_wvalid), .s_ready(m_axi_wready),
    .s_payload({m_axi_wlast, m_axi_wstrb, m_axi_wdata}),
    .m_valid(ntrpu_mem_wvalid), .m_ready(ntrpu_mem_wready), .m_payload()
  );
  ntrpu_sim_ready_gate #(.WIDTH(515+C_M_AXI_ID_WIDTH), .PERIOD(13), .BLOCKED(7), .CHANNEL("R")) ntrpu_r_gate (
    .clk(user_clk), .resetn(user_resetn),
    .s_valid(ntrpu_mem_rvalid), .s_ready(ntrpu_mem_rready),
    .s_payload({m_axi_rid, m_axi_rlast, m_axi_rresp, m_axi_rdata}),
    .m_valid(m_axi_rvalid), .m_ready(m_axi_rready), .m_payload()
  );
  initial $display("NTRPU_PCIE_STALLS version=1 channels=W,R");
'''
    if text.count("endmodule") != 1:
        raise ValueError("Unexpected example module count")
    text = text.replace("endmodule", insertion + "\nendmodule")
    if all_channels:
        declarations = "\n".join("  wire ntrpu_mem_" + name + ";" for name in
                                 ("awvalid", "awready", "arvalid", "arready", "bvalid", "bready"))
        text = text.replace(anchor, declarations + "\n" + anchor)
        for name in ("awvalid", "awready", "arvalid", "arready", "bvalid", "bready"):
            pattern = r"(\.s_axi_" + name + r"\s*\()m_axi_" + name + r"(\))"
            text, count = re.subn(pattern, r"\g<1>ntrpu_mem_" + name + r"\2", text)
            if count != 1:
                raise ValueError("Unexpected application port: " + name)
        control = ""
        for channel, delay in (("AW", 7), ("AR", 11), ("B", 13)):
            name = channel.lower()
            if channel == "B":
                width, payload = "C_M_AXI_ID_WIDTH+2", "m_axi_bid, m_axi_bresp"
                source, sink = "ntrpu_mem_b", "m_axi_b"
            else:
                width = "C_M_AXI_ID_WIDTH+C_M_AXI_ADDR_WIDTH+21"
                payload = ", ".join("m_axi_" + name + field for field in
                                    ("id", "addr", "len", "size", "burst", "lock", "cache", "prot"))
                source, sink = "m_axi_" + name, "ntrpu_mem_" + name
            control += f'''
  ntrpu_sim_ready_gate #(.WIDTH({width}), .OFFER_DELAY({delay}), .CHANNEL("{channel}")) ntrpu_{name}_gate (
    .clk(user_clk), .resetn(user_resetn),
    .s_valid({source}valid), .s_ready({source}ready), .s_payload({{{payload}}}),
    .m_valid({sink}valid), .m_ready({sink}ready), .m_payload()
  );
'''
        text = text.replace("endmodule", control + "\nendmodule")
        text = text.replace("NTRPU_PCIE_STALLS version=1 channels=W,R",
                            "NTRPU_PCIE_STALLS version=2 channels=AW,W,B,AR,R aw_delay=7 ar_delay=11 b_delay=13")
    path.write_text(text)
    meta = {"simulation_only": True, "file": str(path.relative_to(root)),
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "W": {"period": 11, "blocked": 6}, "R": {"period": 13, "blocked": 7},
            "scope": "W producer XDMA; R producer BRAM; AW/AR/B unchanged; no reset injection"}
    if all_channels:
        meta.update(AW={"offer_delay": 7}, AR={"offer_delay": 11}, B={"offer_delay": 13},
                    scope="All five AXI channels independently stalled; AW/AR every offer delayed; B from BRAM delayed. No reset injection.")
    (root / "NTRPU_SIM_STALLS.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--all-channels", action="store_true")
    args = parser.parse_args()
    patch(args.example, args.all_channels)
