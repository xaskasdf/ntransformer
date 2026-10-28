"""Check exported correspondence evidence; do not turn export success into power qualification."""
import argparse,csv,hashlib,json,re
from pathlib import Path

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def table(path,columns):
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t');require(reader.fieldnames==columns,'Inventory columns: '+path.name)
        rows=list(reader)
    require(rows and all(None not in r and None not in r.values() and r[columns[0]] for r in rows),'Malformed inventory')
    require(len({r[columns[0]] for r in rows})==len(rows),'Duplicate inventory name')
    return rows

def review(run,source_sha,inputs):
    run=Path(run)
    result=dict(run=run.name,source_sha256=source_sha,errors=[],windows=[],pending=[],
        ready_for_power_estimate=False,power_estimate_executed=False,board_power_qualified=False,exact_dimm_qualified=False)
    try:
        inv=json.loads((run/'invocation.json').read_text())
        require(inv.get('exit_code')==0 and inv.get('finished_utc'),'Mapping run not successful and terminal')
        require(inv.get('source_sha256')==source_sha and inv.get('inputs')==inputs,'Unbound inputs')
        require(inv.get('frozen_source_unchanged') is True and inv.get('external_inputs_unchanged') is True,'Input changed')
        require(inv.get('power_estimate_executed') is False,'Unexpected power estimate')
        console=(run/'console.log').read_text(errors='replace')
        require(console.splitlines().count('NTRPU_DDR_SAIF_MAPPING_R1_EXPORTED')==1,'Missing/duplicate export')
        errors=[s for s in console.splitlines() if re.match(r'\s*(?:ERROR:|FATAL:|Fatal:)',s)]
        require(not errors,'Native diagnostics: '+repr(errors[:5]))
        result['warnings']=[s for s in console.splitlines() if s.startswith(('WARNING:','CRITICAL WARNING:'))]
        inventories=[]
        for window in range(2):
            name='prbs_'+str(window);root=run/'output'/name
            require(console.splitlines().count('NTRPU_DDR_SAIF_MAPPING_WINDOW_EXPORTED window='+str(window))==1,'Window marker')
            pairs=[line.split('\t') for line in (root/'identity.tsv').read_text().splitlines()]
            require(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'Identity duplicate/malformed')
            identity=dict(pairs)
            require(identity.get('TOP')=='example_top' and identity.get('NAME') and identity.get('PART')=='xcvu5p-flva2104-2-i','Design identity')
            require(identity.get('PORTS')=='154' and identity.get('WINDOW')==name and identity.get('VIVADO')=='2026.1','Window/tool identity')
            require(identity.get('STRIP_PATH')=='ntrpu_ddr_example_activity_tb/dut','Wrong SAIF prefix')
            nets=table(root/'nets.tsv',['name','type']);ports=table(root/'ports.tsv',['name','direction','nets'])
            cells=table(root/'cells.tsv',['name','ref_name','primitive'])
            require(len(ports)==154 and all(p['direction'] in ('IN','OUT','INOUT') for p in ports),'Port inventory')
            reports=('clocks.rpt','route_status.rpt','utilization.rpt','activity_inherited.rpt','activity_reset.rpt',
                     'annotation.rpt','unmatched.rpt','activity_annotated.rpt','port_activity.rpt','scope.txt')
            for leaf in reports:require((root/leaf).is_file() and (root/leaf).stat().st_size>0,'Missing/empty '+leaf)
            scope=(root/'scope.txt').read_text()
            require('no report_power executed' in scope and 'NOT_POWER_QUALIFIED; NOT_DIMM_QUALIFIED; NOT_BOARD_QUALIFIED' in scope,'Missing scope')
            # Source tags are observations, not a percentage: a net may contain
            # several tags. Preserve reports for the native-format interpretation.
            tags={}
            for leaf in ('activity_inherited.rpt','activity_reset.rpt','activity_annotated.rpt','port_activity.rpt'):
                report=(root/leaf).read_text(errors='replace')
                tags[leaf]={tag:len(re.findall(r'\('+tag+r'\)',report)) for tag in ('A','C','D','S')}
            inventories.append(dict(nets=sorted(nets,key=lambda r:r['name']),ports=sorted(ports,key=lambda r:r['name']),cells=sorted(cells,key=lambda r:r['name'])))
            result['windows'].append(dict(name=name,identity=identity,nets=len(nets),ports=len(ports),cells=len(cells),
                source_tag_occurrences=tags,coverage_percentage=None,saif_sha256=inputs[name]['sha256'],
                files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
        require(inventories[0]==inventories[1],'Different designs between windows')
        result['same_design_inventory']=True
        require(not list((run/'output').rglob('power.*')),'Unexpected power report')
    except (ValueError,OSError,KeyError,TypeError) as exc:result['errors'].append(str(exc))
    result['pending']=['Interpret native annotation summary and unmatched nets',
        'Classify X/Z against exported net/cell/port inventory',
        'Resolve source equivalence including vendor libraries and glbl',
        'Establish representativeness; rank trace and refresh review remain separate',
        'Thermal and electrical conditions required before power estimation']
    result['status']='DIAGNOSTIC_DDR_SAIF_MAPPING_R1' if result['errors'] else 'DDR_SAIF_MAPPING_R1_EXPORTED_WITH_OPEN_ITEMS'
    result['reviewer_sha256']=sha(Path(__file__))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--source-sha',required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=review(a.run,a.source_sha,json.loads(a.inputs.read_text())['inputs'])
    with a.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],errors=report['errors'])))
    raise SystemExit(bool(report['errors']))
