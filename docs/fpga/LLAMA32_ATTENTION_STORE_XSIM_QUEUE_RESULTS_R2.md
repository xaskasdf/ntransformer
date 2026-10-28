# Escritor de activaciones: XSim aprobado y continuidad de cola (R2)

Estado 20261013T214901Z: **escritor AXI aprobado y respaldado**. La composición
con el cuantizador HLS real comenzó en `20261013T214728Z-V7-llama32-attention-quantize-store-vivado-r2`; su resultado sigue pendiente.

## Validación formal del escritor

Tanda `20261013T213827Z-V7-llama32-attention-activation-store-vivado-r1`, RTL real del escritor y BFM de memoria AXI,
cuatro vectores ABI1 de atención, tres semillas y banco de agotamiento:

- 214 escenarios, 196 completions, 193 ACK de fuente y 23 resets.
- 65.170 lecturas; 8.132 AW, 8.132 W y 8.126 B.
- Seis transferencias W quedan sin B por reset coordinado; tres escenarios
  adicionales interrumpen un AW antes de W. No se publica una región parcial.
- 113 regiones válidas al cierre: **462.848 bytes exactos**.
- 123 corrupciones de traza rechazadas: 32 por semilla y 27 en agotamiento.

Síntesis OOC VU5P a 5 ns, parámetros de producción: **1.285 LUT, 1.931 FF**,
sin BRAM/DSP, cajas negras o latches. WNS +1,863 ns; WHS +0,066 ns.
No hay extremos internos sin restricciones; 596 entradas y 800 salidas
carecen de presupuesto de E/S. No es timing enrutado ni DDR física.

Archivo SHA-256 `c6cac10cf0c1f9913cca737cdb6ba1ef0cd9c3d1f41f7504b003805810bee15c`;
revisión `6febafa592ad08b94b37b002c8d2821d969a39142a5c1d9ee6469ab1dd27445c`. Ambos verificados en D: y NFS.
El [contrato y pruebas nativas](LLAMA32_ATTENTION_ACTIVATION_STORE_R1.md)
siguen vigentes; esta tanda agrega XSim y síntesis.

## Recuperación del coordinador de integración

El observador R1 terminó antes de crear una ejecución remota: leyó el archivo
compartido `ffn-supervised-next.json` durante una escritura directa y recibió
JSON vacío. Se verificó que su proceso había terminado. No se detuvo ni
repitió ninguna validación Vivado por este incidente.

`tools/ntrpu_queue_state_io_r1.py` reintenta durante un plazo acotado las
lecturas parciales de escritores antiguos. Un JSON permanentemente malformado
continúa produciendo error. Los nuevos registros compartidos se guardan en
un archivo temporal completo y se reemplazan atómicamente.

Cuatro pruebas pasaron: escritura parcial recuperable, corrupción persistente,
denegación de lectura transitoria y reemplazo con un lector activo. Comando:
`python -X utf8 tests/test_ntrpu_queue_state_io_r1.py -v`.

La revisión R2 usa **los mismos RTL, banco, vectores, verificador y límite
de 5.400 s**; cambia únicamente la coordinación y su evidencia de recuperación.
Paquete `20261013T214631Z-llama32-attention-quantize-store-vivado-r2`, SHA-256 `5ab0768d4ca8260b9561d2a4c9502058d75b3343e9bee9131a2c51c9061f2c29`;
fuente ZIP `c378f81e0d7b1b373788c71d0b20aa67c66e1777cdb4265c7c1b557a33f978a3`, 63 miembros verificados.
El historial R1 permanece intacto. Auditoría vigente de las 22 etapas: R12.

Después de esta composición faltan el enlace con la raíz RTL, la proyección O
con GEMV real, residual/FFN integradas y la validación de memoria y tarjeta.
