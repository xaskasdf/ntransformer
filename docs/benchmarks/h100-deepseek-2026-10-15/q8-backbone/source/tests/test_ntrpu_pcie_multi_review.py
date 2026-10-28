"""Log-review regressions; these do not simulate or qualify the PCIe core."""
import importlib.util
import pathlib
import unittest

PATH = pathlib.Path(__file__).resolve().parents[1] / "tools/check_ntrpu_pcie_multi.py"
SPEC = importlib.util.spec_from_file_location("multi_review", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

VALID = """NTRPU_PCIE_MULTI_FORMAT version=1 time_unit=ps payload=768
NTRPU_MULTI_CONFIG ep_mps=256 ep_mrrs=256 devctl=0000103f
Transaction Link Is Up...
Check Max Link Speed = 8.0GT/s - PASSED
Check Negotiated Link Width = 5'h10 - PASSED
NTRPU_CQ_REQUEST time_ps=100000 type=0 address=0000000000000100 bytes=32
NTRPU_CQ_REQUEST time_ps=110000 type=0 address=0000000000000400 bytes=256
NTRPU_CQ_REQUEST time_ps=120000 type=0 address=0000000000000500 bytes=256
NTRPU_CQ_REQUEST time_ps=130000 type=0 address=0000000000000600 bytes=256
NTRPU_MULTI_H2C_BYTES_OBSERVED
NTRPU_CQ_REQUEST time_ps=250000 type=0 address=0000000000000300 bytes=32
NTRPU_CQ_REQUEST time_ps=310000 type=1 address=0000000000000800 bytes=256
NTRPU_CQ_REQUEST time_ps=340000 type=1 address=0000000000000900 bytes=256
NTRPU_CQ_REQUEST time_ps=370000 type=1 address=0000000000000a00 bytes=256
NTRPU_MULTI_C2H_BYTES_OBSERVED
NTRPU_CQ_COMPLETE c2h_bytes=768 c2h_packets=3 h2c_bytes=768 h2c_requests=3 descriptors=2 beats=20 stalls=6 first_c2h_ps=310000 last_c2h_ps=400000
NTRPU_PCIE_MULTI_COMPLETE wbytes=768 rbytes=768 wbeats=12 rbeats=12 aw=3 ar=3 b=3 w_stalls=0 r_stalls=0 clock_ps=4000 link_ps=90000 first_w_ps=150000 last_w_ps=200000 first_r_ps=280000 last_r_ps=330000 rp_mps_raw=3 rp_mrrs_raw=2
"""


def host64_log():
    text = "\n".join(line for line in VALID.splitlines() if not line.startswith("NTRPU_CQ_REQUEST")) + "\n"
    text = text.replace("NTRPU_PCIE_MULTI_FORMAT version=1 time_unit=ps payload=768",
                        "NTRPU_PCIE_HOST64_FORMAT version=1 time_unit=ps payload=1536")
    for old, new in [("bytes=768", "bytes=1536"), ("c2h_packets=3", "c2h_packets=8"),
                     ("h2c_requests=3", "h2c_requests=8"), ("descriptors=2", "descriptors=4"),
                     ("beats=20", "beats=44"), ("beats=12", "beats=24")]:
        text = text.replace(old, new)
    text += ("NTRPU_HOST_MEMORY_INITIALIZED regions=8 descriptors=4 payload=1536 address_bits=64\n"
             "NTRPU_HOST64_DESCRIPTORS direction=H2C count=2\n"
             "NTRPU_HOST64_DESCRIPTORS direction=C2H count=2\n"
             "NTRPU_HOST_MEMORY_COMPLETE read_bytes=1664 written_bytes=1536 regions=8 descriptors=4\n")
    for high in range(1, 5):
        descriptor = (high << 32) + (0x100 if high < 3 else 0x300)
        text += f"NTRPU_CQ_REQUEST time_ps=100000 type=0 address={descriptor:016x} bytes=32\n"
        for offset, size in [(0, 64), (64, 256), (320, 256), (576, 192)]:
            address = (high << 32) + 0xfc0 + offset
            text += f"NTRPU_CQ_REQUEST time_ps=120000 type={0 if high < 3 else 1} address={address:016x} bytes={size}\n"
    return text


class ReviewTests(unittest.TestCase):
    def test_complete_log(self):
        self.assertEqual(MODULE.review(VALID)["status"], "PASS")

    def test_truncated_and_duplicate_records(self):
        lines = VALID.splitlines()
        for index in range(len(lines)):
            with self.subTest(line=index, mode="missing"):
                text = "\n".join(lines[:index] + lines[index+1:])
                self.assertEqual(MODULE.review(text)["status"], "FAIL")
        for line in lines[-2:]:
            self.assertEqual(MODULE.review(VALID + line + "\n")["status"], "FAIL")

    def test_numeric_and_address_mutations(self):
        changes = [("c2h_packets=3", "c2h_packets=1"), ("wbytes=768", "wbytes=767"),
                   ("clock_ps=4000", "clock_ps=4000000"), ("b=3", "b=2"),
                   ("rbeats=12", "rbeats=11"), ("descriptors=2", "descriptors=3"),
                   ("address=0000000000000900", "address=0000000000000800"),
                   ("last_r_ps=330000", "last_r_ps=500000"),
                   ("c2h_bytes=768", "c2h_bytes=x")]
        for old, new in changes:
            with self.subTest(change=new):
                self.assertEqual(MODULE.review(VALID.replace(old, new))["status"], "FAIL")

    def test_late_errors_do_not_preserve_pass(self):
        for diagnostic in ["Fatal: bad data", "ERROR bad response", "NTRPU_PCIE_MULTI_INCOMPLETE",
                           "4[108884000] : Received an invalid frame"]:
            self.assertEqual(MODULE.review(VALID + diagnostic)["status"], "FAIL")

    def test_vivado_echo_is_distinct_from_executed_error(self):
        console = ('## if {$bad_input} { error "Invalid input" }\n' + VALID +
                   'Review numerical/status markers and error logs; this marker alone is not a PASS.\n')
        result = MODULE.review(console)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["excluded_tcl_echo_lines"], 1)
        self.assertEqual(result["excluded_informational_notes"], 1)
        self.assertEqual(MODULE.review(console + 'ERROR: Invalid input')["status"], "FAIL")

    def test_stall_coverage(self):
        stalled = VALID.replace("w_stalls=0", "w_stalls=15") + (
            "NTRPU_PCIE_STALLS version=1 channels=W,R\n"
            "NTRPU_AXI_GATE_COMPLETE channel=W beats=12 source_stalls=15 sink_stalls=0\n"
            "NTRPU_AXI_GATE_COMPLETE channel=R beats=12 source_stalls=14 sink_stalls=0\n")
        self.assertEqual(MODULE.review(stalled, require_stalls=True)["status"], "PASS")
        for old, new in [("source_stalls=15", "source_stalls=14"),
                         ("source_stalls=14", "source_stalls=0"),
                         ("channel=R beats=12", "channel=R beats=11"),
                         ("NTRPU_PCIE_STALLS version=1 channels=W,R", "")]:
            self.assertEqual(MODULE.review(stalled.replace(old, new))["status"], "FAIL")

    def test_stall_run_cannot_pass_as_unstalled(self):
        self.assertEqual(MODULE.review(VALID, require_stalls=True)["status"], "FAIL")

    def test_host64_chain_and_page_split(self):
        text = host64_log()
        self.assertEqual(MODULE.review(text, require_host64=True)["status"], "PASS")
        changes = [("0000000200000fc0", "0000000100000fc0"),
                   ("0000000100000100", "0000000000000100"),
                   ("bytes=64", "bytes=128"), ("read_bytes=1664", "read_bytes=1536"),
                   ("direction=H2C count=2", "direction=H2C count=1"),
                   ("c2h_packets=8", "c2h_packets=7")]
        for old, new in changes:
            with self.subTest(change=new):
                self.assertEqual(MODULE.review(text.replace(old, new), require_host64=True)["status"], "FAIL")
        for line in text.splitlines():
            if line.startswith("NTRPU_HOST"):
                self.assertEqual(MODULE.review(text.replace(line+"\n", ""), require_host64=True)["status"], "FAIL")
        self.assertEqual(MODULE.review(VALID, require_host64=True)["status"], "FAIL")

    def test_all_five_channels_and_response_ids(self):
        text = host64_log().replace("w_stalls=0", "w_stalls=30") + (
            "NTRPU_PCIE_STALLS version=2 channels=AW,W,B,AR,R aw_delay=7 ar_delay=11 b_delay=13\n"
            "NTRPU_AXI_GATE_COMPLETE channel=W beats=24 source_stalls=30 sink_stalls=0\n"
            "NTRPU_AXI_GATE_COMPLETE channel=R beats=24 source_stalls=20 sink_stalls=0\n"
            "NTRPU_AXI_GATE_COMPLETE channel=AW beats=3 source_stalls=21 sink_stalls=0\n"
            "NTRPU_AXI_GATE_COMPLETE channel=AR beats=3 source_stalls=33 sink_stalls=0\n"
            "NTRPU_AXI_GATE_COMPLETE channel=B beats=3 source_stalls=39 sink_stalls=0\n"
            "NTRPU_AXI_CONTROL_COMPLETE aw_stalls=21 ar_stalls=33 b_sink_stalls=0 bid_checked=3 rid_beats_checked=24\n")
        result = MODULE.review(text, require_all_stalls=True, require_host64=True)
        self.assertEqual(result["status"], "PASS", result["errors"])
        changes = [("source_stalls=21", "source_stalls=20"), ("source_stalls=33", "source_stalls=32"),
                   ("source_stalls=39", "source_stalls=38"), ("aw_stalls=21", "aw_stalls=22"),
                   ("ar_stalls=33", "ar_stalls=34"), ("b_sink_stalls=0", "b_sink_stalls=1"),
                   ("bid_checked=3", "bid_checked=2"), ("rid_beats_checked=24", "rid_beats_checked=23"),
                   ("channel=B beats=3", "channel=B beats=2"), ("aw_delay=7", "aw_delay=6")]
        for old, new in changes:
            with self.subTest(change=new):
                self.assertEqual(MODULE.review(text.replace(old, new), require_all_stalls=True)["status"], "FAIL")
        for line in text.splitlines():
            if line.startswith(("NTRPU_AXI_GATE_COMPLETE", "NTRPU_AXI_CONTROL_COMPLETE", "NTRPU_PCIE_STALLS")):
                self.assertEqual(MODULE.review(text.replace(line+"\n", ""), require_all_stalls=True)["status"], "FAIL")
                self.assertEqual(MODULE.review(text+line+"\n", require_all_stalls=True)["status"], "FAIL")
        self.assertEqual(MODULE.review(host64_log(), require_all_stalls=True)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
