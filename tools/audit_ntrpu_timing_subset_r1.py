"""Parse and compare Vivado 2026.1 routed timing subsets without accepting timing.

Endpoint sets must come from an independently bound primitive/port inventory.
This checks report consistency, not whether Vivado modelled the hardware correctly.
"""
from decimal import Decimal
import re


class TimingReportError(ValueError):
    pass


def one(pattern, text, label):
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
        raise TimingReportError(f"Expected exactly one {label}; found {len(matches)}")
    return matches[0]


def parse_report(text, *, kind, top, sources, destinations,
                 source_registers=True, destination_registers=True,
                 required_paths=None):
    if kind not in ("setup", "hold"):
        raise TimingReportError("Unknown timing kind")
    if not sources or not destinations:
        raise TimingReportError("Empty allowed endpoint inventory")
    header = text.split("\nSlack (", 1)[0]
    if one(r"^\| Design\s*:\s*(\S+)\s*$", header, "design") != top:
        raise TimingReportError("Wrong design")
    if one(r"^\| Device\s*:\s*(\S+)\s*$", header, "device") != "xcvu5p-flva2104":
        raise TimingReportError("Wrong device")
    if one(r"^\| Design State\s*:\s*(\S+)\s*$", header, "design state") != "Routed":
        raise TimingReportError("Expected routed report")
    if one(r"^\| Temperature Grade\s*:\s*(\S+)\s*$", header, "temperature grade") != "I":
        raise TimingReportError("Wrong temperature grade")
    if not re.fullmatch(r"-2\s+PRODUCTION\s+.+", one(r"^\| Speed File\s*:\s*(.+?)\s*$", header, "speed file")):
        raise TimingReportError("Wrong speed grade")
    if not one(r"^\| Tool Version\s*:\s*(.+?)\s*$", header, "tool version").startswith("Vivado v.2026.1 "):
        raise TimingReportError("Unexpected Vivado version")
    command = one(r"^\| Command\s*:\s*(.+?)\s*$", header, "command")
    expected_delay = "max" if kind == "setup" else "min"
    if not command.startswith("report_timing ") or re.findall(r"(?:^|\s)-delay_type\s+(\S+)", command) != [expected_delay]:
        raise TimingReportError("Wrong report timing command")
    if re.findall(r"(?:^|\s)-max_paths\s+(\S+)", command) != ["20"]:
        raise TimingReportError("Unexpected max_paths")
    starts = list(re.finditer(r"^Slack \(", text, re.M))
    if not starts or len(starts) > 20:
        raise TimingReportError("Missing or excessive timing paths")
    if required_paths is not None and len(starts) != required_paths:
        raise TimingReportError("Wrong timing path count")
    number = r"-?\d+\.\d+"
    paths = []
    for at, start in enumerate(starts):
        end = starts[at + 1].start() if at + 1 < len(starts) else len(text)
        block = text[start.start():end]
        status, slack = one(r"^Slack \((MET|VIOLATED)\)\s*:\s*("+number+r")ns\b.*$", block, "slack")
        source = one(r"^  Source:\s+(\S+)\s*$", block, "source")
        destination = one(r"^  Destination:\s+(\S+)\s*$", block, "destination")
        group = one(r"^  Path Group:\s+(\S+)\s*$", block, "path group")
        if group != "pcie_user_clk":
            raise TimingReportError("Unexpected path clock")
        path_type = one(r"^  Path Type:\s+(.+?)\s*$", block, "path type")
        expected_corner = "Max at " if kind == "setup" else "Min at "
        if expected_corner not in path_type or (kind == "setup" and "Hold" in path_type) or (kind == "hold" and "Setup" in path_type):
            raise TimingReportError("Wrong setup/hold path type")
        for endpoint, allowed, register in [(source, sources, source_registers),
                                             (destination, destinations, destination_registers)]:
            if register:
                if "/" not in endpoint or endpoint.rsplit("/", 1)[0] not in allowed:
                    raise TimingReportError("Register endpoint outside inventory: "+endpoint)
                if endpoint.rsplit("/", 1)[1] not in ("C", "D", "CE", "R", "S"):
                    raise TimingReportError("Unexpected register pin: "+endpoint)
            elif endpoint not in allowed:
                raise TimingReportError("Port endpoint outside inventory: "+endpoint)
        requirement = one(r"^  Requirement:\s+("+number+r")ns\b.*$", block, "requirement")
        if Decimal(requirement) != (Decimal("4") if kind == "setup" else Decimal("0")):
            raise TimingReportError("Wrong single-clock requirement")
        delay, logic, logic_pct, route, route_pct = one(
            r"^  Data Path Delay:\s+("+number+r")ns\s+\(logic ("+number+r")ns \(("+number+r")%\)\s+route ("+number+r")ns \(("+number+r")%\)\)\s*$",
            block, "data path delay")
        levels, primitives = one(r"^  Logic Levels:\s+(\d+)\s*(.*?)\s*$", block, "logic levels")
        if abs(Decimal(delay) - Decimal(logic) - Decimal(route)) > Decimal("0.002"):
            raise TimingReportError("Logic and route delays disagree with total")
        if abs(Decimal(logic_pct) + Decimal(route_pct) - Decimal("100")) > Decimal("0.002"):
            raise TimingReportError("Delay fractions do not sum to 100 percent")
        footer_required = one(r"^\s+required time\s+("+number+r")\s+.*$", block, "footer required time")
        footer_arrival = one(r"^\s+arrival time\s+("+number+r")\s+.*$", block, "footer arrival time")
        footer_slack = one(r"^\s+slack\s+("+number+r")\s+.*$", block, "footer slack")
        if Decimal(footer_slack) != Decimal(slack):
            raise TimingReportError("Header and footer slacks differ")
        if abs(Decimal(footer_required) + Decimal(footer_arrival) - Decimal(slack)) > Decimal("0.002"):
            raise TimingReportError("Slack does not match rounded arrival/required times")
        if Decimal(slack) and (Decimal(slack) < 0) != (status == "VIOLATED"):
            raise TimingReportError("Slack sign and MET/VIOLATED status differ")
        paths.append(dict(source=source, destination=destination, kind=kind, status=status,
            slack_ns=slack, path_group=group, path_type=path_type, requirement_ns=requirement,
            data_delay_ns=delay, logic_delay_ns=logic, route_delay_ns=route,
            logic_percent=logic_pct, route_percent=route_pct, logic_levels=int(levels),
            logic_primitives=primitives, footer_required_ns=footer_required, footer_arrival_ns=footer_arrival))
    slacks = [Decimal(path["slack_ns"]) for path in paths]
    if slacks != sorted(slacks):
        raise TimingReportError("Paths are not ordered by worst slack")
    return dict(kind=kind, top=top, path_count=len(paths), worst_slack_ns=str(min(slacks)), paths=paths)


def compare_readback(original, reopened, **options):
    before = parse_report(original, **options)
    after = parse_report(reopened, **options)
    if before != after:
        raise TimingReportError("Original and reopened path measurements differ")
    return before
