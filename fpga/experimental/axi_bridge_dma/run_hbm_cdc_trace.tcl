open_vcd $::env(HBM_CDC_TRACE)
log_vcd /tb_axi_bridge_hbm_cdc/*
run all
close_vcd
quit
