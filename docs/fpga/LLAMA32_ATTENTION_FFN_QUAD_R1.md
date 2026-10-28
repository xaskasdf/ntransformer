# Comparación de cuatro MAC para atención y FFN

Fecha: 2026-10-16 10:48 UTC. Estado: **compilación de interfaces aprobada;
síntesis y aritmética Vivado encoladas**. No hay mejora medida todavía.

## Hipótesis y referencia

La [referencia real de atención/FFN/residual](LLAMA32_ATTENTION_FFN_REAL_R1.md)
terminó en 22.282.102 ciclos. Gate, up y down consumieron 19.281.163 ciclos
(86,53 %). Esta tanda compara cuatro MAC por motor con los dos de referencia,
tanto en la proyección O como en FFN. No cambia los pesos, los seis bloques
HLS ni las 139 condiciones del banco aprobado.

Se reutilizan los dos módulos de cuatro MAC archivados en
`20261003T223550Z-V7-ffn-quad-brandon`, byte a byte, bajo
`fpga/rtl/attention_ffn_quad_r1/`. La variante es independiente del experimento
de cancelación registrada y no modifica la capa completa ya encolada.

## Comprobación realizada

`20261016T104706Z-V7-attention-ffn-quad-native-r1`: Icarus compiló 65 fuentes
en CPU 2–3, con límite de 120 segundos. La jerarquía contiene dos motores
quad, ocho MAC y ningún motor dual. No se ejecutó el modelo generado:
las seis firmas HLS son sustitutos para comprobar interfaces. Las dos copias
de compatibilidad sólo separan declaraciones ANSI para Icarus; Vivado usa
los originales. El resultado está respaldado y restaurado en D: y NFS.

## Ejecución y criterios

Coordinador `queue_attention_ffn_quad_vivado_r1.py`, PID 22664, después de
`post-ffn-cancel-vivado-r1`. Espera su aprobación y respaldo restaurado,
además de la referencia aritmética y la compilación anterior. Usa el slot
único, CPU 0–1, ocho horas máximas para síntesis y XSim, sin reintento.

- Sintetizar 181 fuentes, 27 IP y seis familias HLS en VU5P, reloj OOC de 5 ns.
- Confirmar identidad del diseño, fuentes compiladas, fixtures y ocho MAC.
- Ejecutar capa 0, muestra 1, token/posición 0, con Q/K/V externos.
- Exigir los mismos 8.192 bytes finales y volúmenes de lectura/escritura.
- Aplicar las 82 alteraciones negativas del revisor aprobado.
- Registrar ciclos totales y por fase, recursos y diferencias WNS/WHS.
- Respaldar y restaurar todo el resultado, incluido un fallo diagnóstico,
  antes de liberar la cola.

Una aprobación numérica no exige mejora de ciclos ni cierre de timing.
La referencia tiene WNS −10,467 ns en síntesis: el reloj solicitado no es
frecuencia física demostrada. Siguen pendientes ruta, DDR/PCIe físicos,
las demás capas, logits y generación completa.

## Identidad y recuperación

Paquete `20261016T104209Z-attention-ffn-quad-vivado-r1`: 307 archivos,
101 vínculos verificados. Configuración y coordinador en
`build/cloud-setup/20261002-board/`.

- Fuente ZIP SHA-256: `12e8e2b89f5833685248746c0210e53eaf5e422020e8d3e529ded59c48871892`.
- Compilación TAR SHA-256: `e9290db1ded0e4648c9d5437ed7ec15f6926157a1be1c98726ca6724914a08c9`.
- Vector final de referencia: `e431a82afd68b25ebc2c6883f0efbbba697d04770a8b030ff49a48c2b9b04123`.

Fuentes y resultados residen en `artifacts/ntrpu-board-vivado/`, enlazado
a D:, y en NFS. El recibo `VERIFIED.json` del handoff
`attention-ffn-quad-handoff-20261016-r1` registra su restauración.
