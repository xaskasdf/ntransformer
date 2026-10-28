# Entrada completa hasta atención: banco aritmético R1

Fecha: 2026-10-16. **Compilado y encolado; aritmética conjunta pendiente.**
Continúa el [wrapper de entrada](LLAMA32_INPUT_ATTENTION_R1.md).

## Diseño bajo prueba

El banco instancia `ntrpu_input_attention_r1` con productor, retención,
consumidor y memoria KV RTL reales. Vivado usará los cuatro HLS originales:
normalización/Q8, MAC Q8, RoPE y atención FP32. La memoria AXI externa es un
modelo del banco; no equivale a DDR física.

Diez asignaciones separadas por 1 TiB comienzan 4.032 bytes después de su
base, ejercitando cruces de página y direcciones de 64 bits. El caché tiene
otra asignación y capacidad de cuatro tokens.

1. Normalización lee el embedding y pesos FP32; publica 512 palabras ABI1
   de 64 bits y las escribe en 64 beats AXI.
2. Q/K/V lee esos bytes después de B exitoso. Usa las matrices Q8_0 completas:
   2.048/512/512 filas y 64 bloques por fila. Escribe 192 beats FP32.
3. RoPE y captura V leen las escrituras reales del productor. ABI1 y Q/K/V
   empiezan inválidas y sólo cambian tras B exitoso; las referencias nunca
   rellenan esas memorias.
4. Append escribe K rotada/V en disposición GQA. Atención lee ese caché.
   El ACK público precede al del consumidor, su drenaje y el del productor.

El banco cambia los descriptores del host tras admitir el comando y añade
esperas en los cuatro maestros AXI. Observa internamente sin forzar estados
ni reiniciar entre trabajos.

## Casos y referencias

Se compararon exactamente 12.288 valores Q/K/V FP32 de cuatro embeddings
—tokens 0, 42, 128000 y 128255— entre las referencias de productor y
consumidor. Se conservan los cálculos independientes de normalización,
RoPE y las 128 referencias de cabeza de atención.

| Trabajo | Posición | Prefijo inicial | Resultado esperado |
| --- | --- | --- | --- |
| 1 | 0 | 0, nuevo caché | Éxito |
| 2 | 1 | 1 | Éxito |
| 3 | 1 | 2 | Rechazo de duplicado antes del productor |
| 4 | 3 | 2 | Rechazo del salto antes del productor |
| 5 | 2 | 2 | Error en última B de KV; prefijo retenido |
| 6 | 2 | 2 | Reintento y éxito |
| 7 | 3 | 3 | Éxito |
| 8 | 0 | 0, nueva vinculación | Éxito |

Cobertura **prevista**, todavía no medida en RTL: seis normalizaciones,
18 matrices, 18.432 resultados MAC FP64, seis RoPE, 160 cabezas y 10.240
palabras públicas. El productor debe aceptar 665.088 beats de lectura y
emitir 1.536 de escritura. Las escrituras parciales del caso fallido KV no
constituyen un prefijo publicado.

El [revisor completo](../../tools/check_ntrpu_input_attention_real_r1.py)
reconstruye memoria desde AW/W/B y verifica las lecturas posteriores.
Revisa cada salida RoPE, 104 palabras de estado por cabeza, salidas de
atención y lectura pública permutada. Conserva la tolerancia de una ULP y
la comparación de estado con el cálculo Decimal de 90 dígitos.

## Evidencia obtenida

- Icarus compiló 54 fuentes: 49 RTL, banco y cuatro sustitutos HLS de
  interfaz **sin ejecutar**. No certifica aritmética.
- Pasaron ocho casos sintéticos del revisor del productor, con 49
  corrupciones rechazadas; la cadena completa rechazó otras 16.
- Compilación y fuentes tienen respaldo y restauración verificados.

Banco: `fpga/tb/tb_ntrpu_input_attention_real_r1.sv`, SHA-256
`6ef98f0f420159c6473896ece19e75ba3b224d8493f549c06b08469e42775989`.
Referencias: manifiesto `04df4a83b7c6ef8cae36904fdc44d54a1e56ca8088e97d9352a892c8597ade68`.
TAR de compilación: `f9c450d9aedb1ee98f219cbdb7adafe61b217c8ddf87d818e56d64671dbfc474`.

## Turno y aprobación pendiente

`input-attention-real-vivado-r1-coordinator.json`, PID 24836, espera la
síntesis/protocolo de `input-attention-vivado-r1` aprobados y restaurados.
Se agregó al final; conserva el turno DDR después de BRAM.

Paquete `20261016T075307Z-input-attention-real-vivado-r1`: 305 archivos,
126 fuentes compilables —49 RTL, 76 Verilog HLS y banco— y 21 IP originales.
ZIP SHA-256 `e328df9204fa84a27cbf01702a28c61b436df1b840907aa0a720d23a0964b369`.
Copias idénticas y restauración verificadas en D: y NFS.

Una semilla completa en XSim, dos CPU y límite de ocho horas, incluida
elaboración. Reutiliza la síntesis exacta anterior. No reintenta automáticamente;
fallo o timeout se archiva y restaura antes de liberar el slot.

La aprobación exige revisar también fuentes compiladas, IP, ausencia de
sustitutos y errores, identidad del paquete y contadores observados.
Siguen pendientes timing físico, DDR real y las 16 capas con proyección O,
FFN, logits y generación. La documentación y evidencia se conservan en
`artifacts/ntrpu-board-vivado/input-attention-real-handoff-20261016-r1/`.
