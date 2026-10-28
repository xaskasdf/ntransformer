"""Reuse the bounded unit runner with a separately named domain-fence bench."""
from pathlib import Path


if __name__=='__main__':
    text=Path(__file__).with_name('run_ntrpu_peer_regions_unit.py').read_text(encoding='utf-8')
    start=text.index("        'fpga/experimental/ntrpu_peer_read_pkg.sv'")
    end=text.index(")]],",start)
    text=text[:start]+"        'fpga/experimental/ntrpu_peer_read_domain.sv','fpga/tb/tb_ntrpu_peer_read_domain.sv'"+text[end:]
    text=text.replace('tb_ntrpu_peer_regions_unit','tb_ntrpu_peer_read_domain')
    text=text.replace('RG ALL_PASS unit cases=8 vectors=278','DM ALL_PASS domains=32')
    text=text.replace('peer_regions_unit','peer_read_domain')
    text=text.replace('RG_WATCHDOG','DM_WATCHDOG').replace('SG_FAIL','DM_FAIL')
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
