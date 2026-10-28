open_vcd $::env(HBM_SOURCE_TRACE)
log_vcd /tb_axi_bridge_hbm_source/*
run all
close_vcd
quit
