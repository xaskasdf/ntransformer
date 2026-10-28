# Llama 3.2 1B: pesos Q8 reales para NTRPU

Fecha: 2026-10-11 UTC. La validación CPU está terminada; el banco RTL queda
preparado para ejecutarse después de las tandas PCIe ya encoladas.

## Artefacto fijado

Se usa [Unsloth Llama-3.2-1B-Instruct Q8_0](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q8_0.gguf),
revisión `b69aef112e9f895e6f98d7ae0949f72ff09aa401`.

- Tamaño: **1.321.082.528 bytes**.
- SHA-256: `3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
- Carpeta local: `D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/`.
- `audit-r1/audit.json` conserva índice, offsets y hashes de los 147 tensores;
  `hub-response.json` y `config.json` fijan la consulta y configuración.

Se verificaron GGUF v3 little-endian, límites/alineación de cada tensor,
113 matrices Q8_0, 34 tensores FP32 y **38.617.088 escalas Q8 finitas**.
El payload suma 1.313.251.456 bytes. Las dimensiones confirman D=2048,
FFN=8192, 16 capas, 32 cabezas Q, 8 KV y vocabulario 128256.
No hay `output.weight`; la configuración declara embeddings compartidos.
`rope_freqs.weight` contiene 32 valores FP32 adicionales a las normas.

El GGUF antiguo de `D:/AI/models/ter-bench/` tiene otro hash y cabecera de
distinto tamaño, pero los 147 payloads coinciden. Esto no certifica igualdad
del tokenizer ni del resto de metadatos. Se conserva sin modificar.

## Banco numérico inicial

`tools/prepare_ntrpu_llama32_gemv_fixture_r1.py` extrae los primeros y últimos
32 renglones de Q/K/V/O/gate/up/down en las capas 0 y 15: **28 jobs**, 896
salidas y 81.920 bloques por semilla. Conserva los bytes Q8_0 originales;
no descuantiza y vuelve a cuantizar los pesos. Usa activaciones sintéticas
INT8 con escalas FP64, no estados ocultos obtenidos de una inferencia.

Las referencias escalar Python y NumPy coinciden bit a bit; otro recorrido
verifica cada posición de peso, descriptor y hueco de la imagen AXI.
Se contrastaron 2.785.280 bytes directamente con sus offsets del GGUF.
El manifiesto del fixture es
`258ffcdad9e2cb1890b46526f67b0f0b92491c0db3dffcffa254c76ab0e038a5`.
El revisor rechazó 12 corrupciones de resultados/evidencia.

## Ejecución y límites

El banco separado `ntrpu_board_llama32_q8_gemv_r1.tcl` reutiliza el GEMV
dual y el MAC HLS respaldado, con semillas 1/7/42: 84 jobs y 2688 salidas
esperadas. El testbench anterior permanece intacto. El controlador
`queue_llama32_q8_gemv_r1.py` espera el cierre revisado y respaldado de
`commit-priority-mask-rows-r1`; no adelanta ni reinicia la cola PCIe.
Cada ejecución tiene snapshot propio y respaldo de resultado o diagnóstico.

El PASS CPU no es PASS RTL. Faltan proyecciones completas, FFN con pesos
reales, RMSNorm, RoPE, atención GQA, KV y comparación de logits/tokens.
Este derivado cuantizado tampoco prueba equivalencia con el checkpoint
Meta BF16 ni acredita timing físico, DDR real o inferencia en hardware.

Para reproducir la auditoría, usar `uv run --no-project --with gguf==0.19.0
python tools/audit_ntrpu_llama32_gguf_r1.py --model ARCHIVO --output NUEVA_CARPETA`.
El exportador acepta `--model`, `--audit` y `--output`; rechaza sobrescrituras.
