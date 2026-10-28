"""Independently reconstruct bounded PCIe FFN transfers and retirement evidence."""
import argparse
from collections import Counter
import json
import hashlib
import re
import struct
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference():
    # Mathematical fixture independent of SV: gate=257, up=x, SiLU(257)=257
    # at FP64 precision, activation Q8 scale=257, down permutes by 7 mod 32.
    q = [127, 1] + [i*7-113 for i in range(2, 32)]
    inputs = [bytes(v & 255 for v in q)+struct.pack('<d', 1.0)+bytes(24)]
    for b in range(1, 4):
        weights = bytearray()
        for row in range(32):
            weights.extend(struct.pack('<e', 257.0 if b == 1 else 1.0))
            values = bytearray(32)
            values[1 if b == 1 else row if b == 2 else (row*7)%32] = 1
            weights.extend(values)
        inputs.append(bytes(weights))
    outputs = [struct.pack('<32d', *([257.0]*32)), struct.pack('<32d', *q),
               bytes(v & 255 for v in q)+struct.pack('<d', 257.0)+bytes(24),
               struct.pack('<32f', *(257*q[(row*7)%32] for row in range(32)))]
    return inputs+outputs


def fixture_reference(path):
    """Read the separately qualified numerical fixture, independent of SV packing."""
    path=Path(path)
    m=json.loads((path/'manifest.json').read_text())
    require((m['format'],m['dim'],m['hidden'],m['layout'],m['tile_k'])==
            ('NTRPCIEFFN01',256,720,2,[4,4,8]), 'qualified Brandon fixture contract')
    require(m['model']=='xaskasdf/brandon-tiny-10m-instruct' and
            m['revision']=='ca6329de991a1cfc17b2a7a1aaae2cc38c08577a' and
            m['weights_sha256']=='2ff727cc8921d21e5e7b153babd288efc5524b166e9417e1b439ac8c11fe44cf',
            'qualified real-weight identity')
    for name,sha in m['files'].items():
        p=(path/name).resolve()
        require(p.is_relative_to(path.resolve()) and hashlib.sha256(p.read_bytes()).hexdigest()==sha,
                'fixture file identity: '+name)
    images=[bytes.fromhex((path/f'buffer-{b}.hex').read_text()) for b in range(8)]
    require(list(map(len,images))==[512,195840,195840,200192,5760,5760,1472,1024], 'reference buffer geometry')
    require(all(hashlib.sha256(image).hexdigest()==m['buffers'][b]['data_sha256'] for b,image in enumerate(images)), 'reference data identities')
    reference_raw=(path/'reference-manifest.json').read_bytes()
    require(hashlib.sha256(reference_raw).hexdigest()==m['reference_manifest_sha256'], 'original FFN reference manifest')
    original=json.loads(reference_raw)
    require(all(original[k]==m[k] for k in ['model','revision','weights_sha256','dim','hidden','layout','tile_k']), 'FFN source reference identity')
    return images,m


def review(text, *, concurrent=False, rx_split=False, fixture=None):
    if fixture is None:
        expected,metadata=reference(),None
        dim,hidden,tile,host_stride=32,32,0x020202,0x10000
    else:
        expected,metadata=fixture_reference(fixture)
        dim,hidden,tile,host_stride=256,720,0x080404,0x100000
    h2c_size,stage_size,output_size=sum(map(len,expected[:4])),sum(map(len,expected[4:])),len(expected[7])
    require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )', text, re.M|re.I), 'simulation error')
    lines = [line for line in text.splitlines() if line.startswith('NTRPU_PCIE_FFN_')]
    events = []
    for line in lines:
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        events.append((line.split()[0].removeprefix('NTRPU_PCIE_FFN_'), fields))
    def selected(name):
        return [v for k, v in events if k == name]
    require(len(selected('PASS')) == 1, 'one final PASS')
    require(selected('PASS')[0] == dict(cases='3', computed='1', rejected='2',
            h2c_bytes=str(h2c_size), core_bytes=str(stage_size), c2h_bytes=str(output_size), msix='2'), 'final accounting')
    require([(int(v['direction']), int(v['count']), int(v['status'], 16)) for v in selected('DMA')] ==
            [(0, 4, 6), (1, 1, 6)], 'DMA descriptor counts and status')
    controls = selected('DMA_CONTROL')
    require([int(v['direction']) for v in controls] == [0, 1] and
            all(int(v['value'], 16) == 0x00fffe7e for v in controls), 'DMA stopped/incrementing mode readback')
    capabilities = {int(v['register'], 16): int(v['value'], 16) for v in selected('MSIX_CAP')}
    require(capabilities == {0x60: 0x801f7011, 0x64: 0x8001, 0x68: 0x8fe1}, 'advertised MSI-X table and PBA')
    drains = selected('DMA_DRAIN')
    require([int(v['direction']) for v in drains] == [0, 1], 'completed DMA phases')
    mode = selected('MODE')
    require(mode == [{'concurrent': str(int(concurrent))}] or not mode and not concurrent, 'requested MMIO/DMA mode')
    require(selected('BARS') == [{'xdma': '1', 'csr': '0'}], 'expected generated BARs')
    base = [0x100000fc0+b*0x20000000 for b in range(8)]
    observed = {}
    writes = Counter()
    ram_read_bytes = 0
    bursts = {'R': None, 'W': None}
    last_time = -1
    for kind, fields in events:
        if 'time_ps' in fields:
            now = int(fields['time_ps'])
            require(now >= last_time, 'chronological event log')
            last_time = now
        if kind in {'RAM_AR', 'RAM_AW'}:
            channel = kind[-1]
            address, beats, ident = int(fields['addr'], 16), int(fields['beats']), int(fields['id'])
            require(bursts[channel] is None, 'overlapping memory burst')
            require(address % 64 == 0 and 1 <= beats <= 256 and (address & 4095)+beats*64 <= 4096,
                    'AXI alignment/page bounds')
            bursts[channel] = [address, beats, ident]
        if kind in {'RAM_R', 'RAM_W'}:
            channel = kind[-1]
            address, ident = int(fields['addr'], 16), int(fields['id'])
            require(bursts[channel] is not None, 'data without memory address')
            require([address, ident] == [bursts[channel][0], bursts[channel][2]], 'AXI address/ID progression')
            data = int(fields['data'], 16).to_bytes(64, 'little')
            region = next((b for b in range(8) if base[b] <= address and address+64 <= base[b]+len(expected[b])), None)
            require(region is not None, 'exact fixture bounds, all address bits')
            off = address-base[region]
            require(data == expected[region][off:off+64], 'independent numerical RAM comparison')
            if channel == 'W':
                for lane, value in enumerate(data):
                    require(address+lane not in observed, 'duplicate RAM write')
                    observed[address+lane] = value
                writes[region] += 64
            else:
                require(bytes(observed.get(address+i, 0xa5) for i in range(64)) == data,
                        'RAM read before matching write')
                ram_read_bytes += 64
            bursts[channel][0] += 64
            bursts[channel][1] -= 1
            if not bursts[channel][1]:
                bursts[channel] = None
    require(all(v is None for v in bursts.values()), 'complete RAM bursts')
    require([writes[b] for b in range(8)] == [len(data) for data in expected], 'all input/intermediate/output bytes written once')
    require([int(v['count']) for v in selected('MSIX')] == [1, 2], 'two MSI-X deliveries')
    require(len(selected('ACK')) == 2, 'two actual XDMA acknowledgements')
    require([int(v['id']) for v in selected('CASE')] == [1, 2, 3], 'three job cases')
    for tag in ('JOB', 'DONE', 'RETIRE'):
        require([(int(v['sequence'], 16), int(v['epoch'], 16)) for v in selected(tag)] == [(1, 601), (2, 602), (3, 603)], tag+' owner identities')
    require([(int(v['status']), int(v['stages']), int(v['bytes']), int(v['stage_bytes'])) for v in selected('DONE')] ==
            [(0, 4, output_size, stage_size), (1, 0, 0, 0), (1, 0, 0, 0)], 'compute and rejected completion metadata')
    irq = selected('IRQ')
    require([int(v['level']) for v in irq] == [1, 0, 1, 0], 'IRQ edges and polling silence')
    times = lambda name: [int(v['time_ps']) for v in selected(name)]
    ack, retire, edges, jobs, done = (times(k) for k in ['ACK', 'RETIRE', 'IRQ', 'JOB', 'DONE'])
    # Driver SERVICE is task return after its posted-write delay; use the actual
    # AXI-Lite OKAY response to time software service instead. The response
    # and IRQ falling edge may be observed on the same host clock edge.
    service = [int(v['time_ps']) for v in selected('MMIO_W') if
               v['addr'] == '00000010' and v['data'] == '00000004' and v['resp'] == '0']
    require(len(service) == 3, 'three accepted software services')
    if concurrent or selected('SERVICE_ACCEPT'):
        require([(int(v['sequence'],16),int(v['epoch'],16)) for v in selected('SERVICE_ACCEPT')] ==
                [(1,601),(2,602),(3,603)], 'notifier accepted software identities')
        # New observer times the source notifier handshake directly; B responses
        # cross a different CDC path and do not timestamp service acceptance.
        service = times('SERVICE_ACCEPT')
    require(done[0] < edges[0] <= ack[0] < service[0] <= edges[1] < retire[0] < jobs[1], 'transport first ordering')
    require(done[1] < edges[2] < service[1] < ack[1] < edges[3] < retire[1] < jobs[2], 'software first ordering')
    require(done[2] < service[2] < retire[2], 'polling retirement')
    reads, mmio_writes = selected('MMIO_R'), selected('MMIO_W')
    require(all(int(v['resp']) == 0 for v in reads), 'all BAR reads OKAY')
    errors = [v for v in mmio_writes if int(v['resp']) != 0]
    require(len(errors) == 1 and errors[0]['addr'] == '00000010' and errors[0]['data'] == '00000004' and errors[0]['resp'] == '2', 'only wrong-owner service rejected')
    shadows = {}
    owner = 0
    for kind, v in events:
        if kind == 'MMIO_W':
            address, data, response = int(v['addr'], 16), int(v['data'], 16), int(v['resp'])
            if address == 0x10 and data == 4:
                identity = (shadows.get(0x38), shadows.get(0x3c))
                require(identity == ((2, 601) if response else (owner, 600+owner)), 'software owner identity')
            if response == 0:
                shadows[address] = data
        elif kind == 'JOB':
            owner = int(v['sequence'], 16)
            config = [owner, 600+owner, 600+owner, hidden<<16|(dim if owner==1 else 0), 2, tile]
            require([shadows.get(0x20+i*4) for i in range(6)] == config, 'admitted descriptor configuration')
            require(shadows.get(0x14) == (owner != 3), 'captured IRQ mode')
            for b in range(8):
                addr, region_base = base[b], base[b]-4032
                words = [addr & 0xffffffff, addr >> 32, region_base & 0xffffffff, region_base >> 32,
                         4032+len(expected[b]), 0, 600+owner]
                require([shadows.get(0x100+b*32+f*4) for f in range(7)] == words, 'registered buffer descriptor')
        elif kind == 'MMIO_R':
            address, data = int(v['addr'], 16), int(v['data'], 16)
            fixed = {0: 0x4652544e, 4: 0x20000, 8: 15}
            if owner:
                fixed.update({0x18: owner, 0x1c: owner, 0x64: owner, 0x68: 600+owner,
                              0x4c: owner, 0x50: 600+owner, 0x44: output_size if owner == 1 else 0,
                              0x48: stage_size if owner == 1 else 0, 0x40: 0x1c00 if owner == 1 else 1,
                              0x54: 0, 0x60: 0})
            if address in fixed:
                require(data == fixed[address], 'BAR readback content')
            if address == 0xc:
                require(not data & 0x80, 'notifier fault remained clear')
    # Decode CQ independently, including exact destination and MSI-X payload.
    c2h = bytearray()
    packet = None
    h2c = Counter()
    desc_reads = 0
    msix = 0
    for kind, v in events:
        if kind == 'CQ_REQUEST':
            require(packet is None, 'no overlapping CQ packet')
            address, count, typ = int(v['addr'], 16), int(v['bytes']), int(v['type'])
            require(0 < count <= 256 and (address & 4095)+count <= 4096, 'CQ page and MPS/MRRS')
            if typ == 0:
                if 0x100000100 <= address and address+count <= 0x100000180 or 0x300000300 <= address and address+count <= 0x300000320:
                    desc_reads += count
                else:
                    region = next((b for b in range(4) if address == 0x200000fc0+b*host_stride+h2c[b] and h2c[b]+count <= len(expected[b])), None)
                    require(region is not None, 'ordered H2C request bounds')
                    h2c[region] += count
            else:
                require(typ == 1, 'CQ request type')
                packet = [address, count, bytearray()]
        elif kind == 'CQ_DATA':
            require(packet is not None, 'CQ data without header')
            address, count, first, left = (int(v[key], 16 if key == 'addr' else 10) for key in ['addr', 'bytes', 'first', 'left'])
            require(address == packet[0] and left == packet[1]-len(packet[2]), 'CQ packet progression')
            require(first == (16 if not packet[2] else 0) and count == min(left, 64-first), 'CQ beat lane boundaries')
            packet[2].extend(int(v['data'], 16).to_bytes(64, 'little')[first:first+count])
            if len(packet[2]) == packet[1]:
                if address == 0x500000800:
                    require(packet[2] == struct.pack('<I', 0x8badc0de), 'independent MSI-X vector data')
                    msix += 1
                else:
                    require(address == 0x400000fc0+len(c2h), 'C2H full address progression')
                    c2h.extend(packet[2])
                packet = None
    require(packet is None and msix == 2 and c2h == expected[7], 'complete PCIe write data')
    require(desc_reads == 160 and [h2c[b] for b in range(4)] == list(map(len,expected[:4])), 'complete DMA source data')
    overlap = []
    logged_packets = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
                      if line.startswith('NTRPU_PCIE_LOG_PACKET ')]
    for direction, desc in enumerate([0x100000100, 0x300000300]):
        first = next(int(v['time_ps']) for v in selected('CQ_REQUEST') if int(v['addr'], 16) == desc)
        drained = int(drains[direction]['time_ps'])
        require(drained > first, 'DMA phase interval')
        # Register access tasks emit these lines on the actual root RQ path.
        requests = re.findall(r'^\[\s*(\d+)\] : Mem(?:32|64) (?:Read|Write) Req @address', text, re.M)
        if concurrent:
            if direction == 0:
                last = max(int(v['time_ps']) for v in selected('RAM_W')
                           if any(start <= int(v['addr'],16) < start+len(expected[b]) for b,start in enumerate(base[:4])))
            else:
                last = max(int(v['time_ps']) for v in selected('CQ_DATA') if
                           0x400000fc0 <= int(v['addr'],16) < 0x400000fc0+output_size)
            packets = [v for v in logged_packets if v['channel']=='0' and
                       # Generated root tasks place Tag[9:8] in DW0 bits 23/19.
                       (int(v['head'],16) & ~0x00880000) in {0x00000001,0x20000001,0x40000001,0x60000001} and
                       first <= int(v['time_ps']) <= last]
            require(bool(packets), 'actual register TLP must overlap DMA data interval')
            overlap.append(dict(direction=direction,first_ps=first,last_ps=last,register_packets=len(packets)))
        else:
            require(not any(first < int(time) < drained for time in requests), 'register traffic overlaps bounded DMA phase')
    if concurrent:
        status = selected('DMA_STATUS')
        require({int(v['direction']) for v in status}=={0,1} and
                all(int(v['value'],16)&~7==0 for v in status), 'DMA status polls without errors')
        require(all([int(v['value'],16) for v in status if int(v['direction'])==direction][-1]==6
                    for direction in (0,1)), 'both DMA engines reach stopped/completed')
        cc = [v for v in logged_packets if v['channel']=='1']
        cq_reads = [v for v in selected('CQ_REQUEST') if v['type']=='0']
        require(len(cc)==len(cq_reads), 'one logged completion per CQ read')
        require([int(v['bytes']) for v in cc]==[int(v['bytes'])+12 for v in cq_reads], 'logged CC byte counts')
        headers = [0x4a000000 | int(v['tc'])<<20 | (int(v['attr'])>>2)<<18 |
                   (int(v['attr'])&3)<<12 | int(v['bytes'])//4 for v in cq_reads]
        require([int(v['head'],16) & ~0x00880000 for v in cc]==headers, 'logged CC header lengths and attributes')
    rx_summary = None
    if rx_split:
        received = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
                    if line.startswith('NTRPU_PCIE_RX_LOG_PACKET ')]
        require(received and all(v['channel'] in {'0','1'} for v in received), 'separate RX log channels')
        cq = [v for v in received if v['channel']=='0']
        requests = selected('CQ_REQUEST')
        require(len(cq)==len(requests), 'one CQ log per independently observed request')
        for logged, actual in zip(cq, requests):
            write = int(actual['type'])==1
            count = int(actual['bytes'])
            require(int(actual['addr'],16)>0xffffffff, 'fixture CQ uses 64-bit addresses')
            header = (0x60000000 if write else 0x20000000) | int(actual['tc'])<<20 | \
                     (int(actual['attr'])>>2)<<18 | (int(actual['attr'])&3)<<12 | count//4
            require(int(logged['bytes'])==16+(count if write else 0) and
                    int(logged['head'],16)==header, 'CQ log length/header agrees with AXIS observer')
            require(int(logged['time_ps'])>=int(actual['time_ps']), 'CQ parsed after first accepted beat')
        # This fixture has only one outstanding host MMIO/config read. Every
        # request produces one RC completion; posted memory writes produce none.
        response_sizes=[]
        for packet in logged_packets:
            if packet['channel']!='0':continue
            head=int(packet['head'],16)&~0x00880000
            if head in {0x00000001,0x20000001,0x04000001}:response_sizes.append(16)
            elif head==0x44000001:response_sizes.append(12)
        rc=[v for v in received if v['channel']=='1']
        require([int(v['bytes']) for v in rc]==response_sizes, 'one correctly sized RC per nonposted host request')
        require(all(int(v['head'],16)==(0x4a000001 if int(v['bytes'])==16 else 0x0a000000)
                    for v in rc), 'RC completion headers')
        rx_summary=dict(cq_packets=len(cq),rc_packets=len(rc),cq_bytes=sum(int(v['bytes']) for v in cq),
                        rc_bytes=sum(response_sizes))
    return dict(cases=3, computed=1, invalid=2, output_fp32=dim, h2c_bytes=h2c_size, core_written_bytes=stage_size,
                c2h_bytes=len(c2h), ram_read_bytes=ram_read_bytes, msix=msix,
                mmio_writes=len(mmio_writes), mmio_reads=len(reads), rejected_mmio=1,
                compute_cycles=(done[0]-jobs[0])//5000,
                event_to_retire_ps=[retire[i]-done[i] for i in range(3)],
                dma_overlap=overlap,
                rx_logging=rx_summary,
                fixture_identity=None if metadata is None else {key:metadata[key] for key in ['model','revision','weights_sha256','reference_manifest_sha256','sample']},
                scope=('Concurrent' if concurrent else 'Serialized')+' MMIO/DMA PIPE simulation with behavioral RAM; no production host driver, DDR or hardware performance claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--concurrent', action='store_true', help='Require actual register TLPs during both DMA data intervals')
    parser.add_argument('--rx-split', action='store_true', help='Audit separate CQ/RC logs against accepted AXIS requests')
    parser.add_argument('--fixture', type=Path, help='Qualified prepared Brandon fixture; omit for synthetic D32 reference')
    args = parser.parse_args()
    result = review(args.log.read_text(), concurrent=args.concurrent, rx_split=args.rx_split, fixture=args.fixture)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
