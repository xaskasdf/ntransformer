# Cola Vivado del secuenciador de 32 cabezas

Encolado el 2026-10-13 a las 18:18 UTC. Padre:
`llama32-attention-engine-vivado-r4-coordinator.json`.
No modifica los bancos existentes ni adelanta una dependencia sin revisión.

El propietario `llama32-attention-head-sequencer-vivado-r1-coordinator.json`
espera PASS del motor generado, sus 155 consultas/155 tiles numéricos,
cuatro resets, 9.600 resultados y 27 controles negativos. Verifica el hash
de la revisión dentro del archivo terminal respaldado antes de usar el slot.

## Trabajo preparado

- Cuatro suites XSim: 35 escenarios cada una; watchdog 32 con semillas
  1/7/42 y watchdog de producción 65.536 con semilla 1.
- Peer de protocolo determinista: revisa orden, leases, Q, publicación,
  cancelación, revocación, errores, watchdog y resets. No ejecuta atención.
- Síntesis OOC de `ntrpu_attention_head_sequencer_r1` a 5 ns en VU5P.
  Se exige BRAM inferida, cero cajas negras/latches y clocks internos válidos.
- Dos CPU, límite de 1.200 s, sin registro automático de ondas.
- Revisión independiente de 73.728 palabras de salida, diez corrupciones
  por suite, hashes de fuentes compiladas, recursos, timing y DCP.
  Un WNS negativo se registra como estimación fallida de timing;
  un PASS funcional/OOC no demuestra timing ruteado.

El controlador deja `cache_lock` para el wrapper pendiente. Su entrada
`cache_committed_tokens` debe provenir de autoridad interna del caché.
La prueba aislada no demuestra que bind/append estén bloqueados en el
motor compuesto; esa integración tiene que implementarse y comprobarse.

## Paquete y respaldo

Paquete de 37 archivos, con RTL/banco, scripts, revisión nativa, controles
de mutación y logs positivos. La tanda nativa rechazó cinco mutantes RTL.

- Fuente: `dc02e774ebdfd96c03337b30d8f54a31542c07e3edb09bfaeca4e95a8e405c95`.
- Paquete: `bc0c03212b7669081b8925d378814de40a3fa3f6827d14bb8dd26b6f958ad612`.
- Driver: `52bfa59ae92e474f536e09cbf25a8d55f12d83159b60950290b179618173b0ef`.
- Local: `D:/AI/ntransformer/artifacts/ntrpu-model-validation/20261013T181735Z-llama32-attention-head-sequencer-vivado-r1/`.
- NFS: directorio `source-20261013T181735Z-llama32-attention-head-sequencer-vivado-r1`
  de la sesión de placa `20261002T085705Z`.

Fuentes y contrato R1 quedan congelados. Una corrección posterior requiere
otra revisión. El estado formal y el archivo terminal decidirán el resultado;
estar encolado no equivale a haber pasado Vivado.
