open_vcd $::env(HBM_CDC_QUEUES_TRACE)
log_vcd /tb_axi_bridge_hbm_cdc_queues/*
run all
close_vcd
quit
