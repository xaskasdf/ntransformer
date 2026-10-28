"""Independent acceptance of a DDR power export; never qualifies board power."""
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET


EXPECTED = {
    "ntrpu_board_ddr_power_probe_r2.tcl": "8bbc24239bebee597a84b094333f1960b6e23790ca285e63aa2f2d92c750be57",
    "route_attempt.dcp": "161d111ce97d471d461328e615c96d32232b6e6c101673de210cb75b282f205c",
}
MARKER = "NTRPU_POWER_DIAGNOSTIC_EXPORTED; NOT_POWER_QUALIFIED"


def review(evidence):
    assert evidence["terminal"] is True
    files = evidence["files"]
    for name, entry in files.items():
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"], name
    text = lambda name: files[name]["text"]
    invocation = json.loads(text("invocation.json"))
    assert invocation == evidence["invocation"]
    assert invocation["exit_code"] == 0 and text("exit_code.txt").strip() == "0"
    assert invocation["finished_utc"] and invocation["input_integrity_verified"] is True
    assert invocation["expected_input_sha256"] == EXPECTED
    integrity = json.loads(text("input-integrity.json"))
    assert integrity["unchanged"] is True
    for key in ("expected", "before", "after", "staged_after"):
        assert integrity[key] == EXPECTED, key
    for name in ("inputs.before.json", "inputs.after.json"):
        assert json.loads(text(name)) == EXPECTED
    assert text("console.log").splitlines().count(MARKER) == 1
    assert text("vivado.log").splitlines().count(MARKER) == 1
    identity = dict(line.split("\t") for line in text("reports/design_identity.tsv").splitlines())
    assert identity["NAME"] and identity["TOP"] == "example_top"
    assert identity["PART"] == "xcvu5p-flva2104-2-i" and identity["PORTS"] == "154"
    for name in ("clocks.rpt", "utilization.rpt", "power.rpt", "power.xml", "scope.txt"):
        assert text("reports/" + name).strip(), name
    scope = text("reports/scope.txt")
    for line in (
        "Scope: one generic DDR4 channel, two slots, not qualified F1",
        "No new activity or thermal constraints applied",
        "Inherited activity plus tool defaults; diagnostic only",
        "Not a four-channel board, PCIe or NTRPU power budget",
        "Vivado: 2026.1",
    ):
        assert line in scope.splitlines(), line
    for report in ("utilization.rpt", "power.rpt"):
        assert re.findall(r"^\|\s*Design\s*:\s*(\S+)\s*$", text("reports/" + report), re.M) == ["example_top"]
    xml = ET.fromstring(text("reports/power.xml"))
    assert len(list(xml.iter())) > 1
    console = text("console.log")
    errors = [line for line in console.splitlines() if line.startswith("ERROR:")]
    assert not errors, errors
    return dict(
        status="PASS_DDR_POWER_DIAGNOSTIC_EXPORTED", identity=identity,
        invocation=invocation, xml_root=xml.tag, xml_elements=len(list(xml.iter())),
        reports_sha256={n: v["sha256"] for n, v in files.items() if n.startswith("reports/")},
        warnings=[line for line in console.splitlines() if line.startswith(("WARNING:", "CRITICAL WARNING:"))],
        power_model_messages=[line for line in console.splitlines() if "[Power " in line],
        errors=errors, board_power_qualified=False, exact_dimm_qualified=False,
    )


def controls(evidence):
    """Corrupt independently observed artifacts, rehashing each changed entry."""
    def set_text(e, name, value):
        e["files"][name] = dict(text=value, sha256=hashlib.sha256(value.encode()).hexdigest())

    rejected = []
    for case in ("exit", "marker", "duplicate_marker", "top", "part", "ports", "scope",
                 "missing_report", "empty_report", "xml", "integrity", "input_sha", "error"):
        e = copy.deepcopy(evidence)
        name = "console.log"
        value = e["files"][name]["text"]
        if case == "exit":
            name, value = "exit_code.txt", "1\n"
        elif case == "marker":
            value = value.replace("\n" + MARKER + "\n", "\n")
        elif case == "duplicate_marker":
            value += "\n" + MARKER + "\n"
        elif case in ("top", "part", "ports"):
            name = "reports/design_identity.tsv"
            identity = dict(line.split("\t") for line in e["files"][name]["text"].splitlines())
            identity[case.upper()] = "wrong"
            value = "".join(k + "\t" + v + "\n" for k, v in identity.items())
        elif case == "scope":
            name, value = "reports/scope.txt", "Board qualified\n"
        elif case == "missing_report":
            del e["files"]["reports/clocks.rpt"]
        elif case == "empty_report":
            name, value = "reports/power.rpt", ""
        elif case == "xml":
            name, value = "reports/power.xml", "<broken>"
        elif case in ("integrity", "input_sha"):
            name = "input-integrity.json"
            data = json.loads(e["files"][name]["text"])
            if case == "integrity":
                data["unchanged"] = False
            else:
                data["after"]["route_attempt.dcp"] = "0" * 64
            value = json.dumps(data)
        elif case == "error":
            value += "\nERROR: injected failure\n"
        if case != "missing_report":
            set_text(e, name, value)
        assert e != evidence
        try:
            review(e)
        except (AssertionError, KeyError, ET.ParseError):
            rejected.append(case)
        else:
            raise AssertionError("Accepted corruption: " + case)
    return rejected
