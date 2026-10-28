"""Expose conservative atomic admission offers in isolated copies of R4."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


OFFER = '''        // Conservative offer: returns alone cannot assert availability.
        // With another free bank, the allocator may prefer a same-edge return.
        // Existing command acceptance consumes the real bank and descriptor.
        wire [BANKS-1:0] offer_free;
        for (genvar ob=0;ob<BANKS;ob=ob+1) begin: offer_bank
            assign offer_free[ob] = !banks_used[ob] && !retired_mask[ob] &&
                ((ob>=2*g && ob<2*g+2) || ob>=2*CHANNELS);
        end
        assign reserve_can[g] = rst_n && !stop && !fault && !allocator_fault &&
            !(|return_error) && !(|response_error) && !fill_active &&
            occupied[g]<DESCRIPTORS && (|offer_free);
'''


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];manifest={}
    for original,target in [('ntrpu_peer_multibuffer_fillstage','ntrpu_peer_multibuffer_offer'),
                            ('ntrpu_peer_multibuffer_elastic','ntrpu_peer_multibuffer_reserved')]:
        source=root/'fpga/rtl'/f'{original}.sv';before=source.read_text();text=before
        text=one(text,'module '+original+' #(','module '+target+' #(')
        text=one(text,'    input wire cmd_valid,','    output wire [CHANNELS-1:0] reserve_can,\n    input wire cmd_valid,')
        if original.endswith('fillstage'):
            text=one(text,'    generate for (g=0; g<CHANNELS; g=g+1) begin: channel_io\n',
                     '    generate for (g=0; g<CHANNELS; g=g+1) begin: channel_io\n'+OFFER)
        else:
            text=one(text,'    ntrpu_peer_multibuffer_fillstage #(','    ntrpu_peer_multibuffer_offer #(')
        dst=args.output/f'{target}.sv';dst.write_text(text)
        (args.output/f'{target}.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile=source.name,tofile=dst.name)))
        manifest[source.name]=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                  generated=dst.name,generated_sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    (args.output/'TRANSFORMATION.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
