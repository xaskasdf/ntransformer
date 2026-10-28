"""Bind the shared-reset V3 profile to the frozen native bridge example."""
import hashlib
import json
from prepare_guard_integration import prepare as prepare_v2_top


def prepare(example,client,output):
    info=prepare_v2_top(example,client,output)
    path=output/'ntrpu_axi_bridge_guard_native.sv';text=path.read_text()
    old='ntrpu_axi_bridge_guarded_client guarded('
    if text.count(old)!=1:raise ValueError('Native binding changed')
    text=text.replace(old,'ntrpu_axi_bridge_guarded_client_v3 #(.SHARED_NATIVE_RESET(1)) guarded(')
    path.write_text(text)
    info.update(status='V3_SHARED_NATIVE_PREPARED',native_top_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shared_reset_contract='Loaded V2 reference proves both outputs share one physical leaf driver; V3 must retain that AXI reset source, and may prune its unused control alias')
    (output/'PREPARED.json').write_text(json.dumps(info,indent=2)+'\n')
    return info
