# Cuantización → memoria AXI → proyección O real, R1

Estado a 20261013T224949Z: **encolado; simulación aritmética integrada y síntesis pendientes**.
La tanda O `20261013T221343Z-V7-llama32-attention-projection-memory-vivado-r1`
continúa sin reinicio. Esta etapa 24 espera a su coordinador canónico R2,
revisión independiente aprobada, archivo verificado y liberación del turno Vivado.

## Qué se conecta

`ntrpu_attention_output_path_r1` conecta el cuantizador HLS real,
su propietario y escritor AXI, y la proyección O con los dos MAC HLS reales.
Se reutilizan fuentes generadas de las tandas calificadas; el paquete contiene
26 módulos Verilog generados y ocho scripts de IP. No incluye sustitutos entre
las fuentes de simulación o síntesis.

Los 2.048 FP32 de entrada provienen de las cuatro muestras congeladas de atención
de Llama 3.2 1B Instruct Q8_0. El banco emula su lectura síncrona; el bloque raíz
de atención aún no está conectado. Los pesos O reales tienen forma 2.048 × 2.048.

El modelo corresponde al enlace del usuario: SHA-256
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
Se conserva en `D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/`.

## Contrato del banco

| Trabajo | Muestra, índice desde cero | Resultado exigido |
| --- | --- | --- |
| 1 | 0 | Cuantización y O completos, comparación exacta |
| 2 | 1 | Cuantización y O completos, comparación exacta |
| 3 | 3 | Cancelación al ofrecer O: cuantización completa, cero lecturas O |
| 4 | 2 | Recuperación: cuantización y O completos, comparación exacta |

Cada cuantización debe producir 512 palabras de 64 bits y publicar 64 registros
ABI1 mediante AXI: 4.096 bytes. La memoria comienza desconocida y solo se llena
con los datos W al aceptar B. O lee esos datos materializados; la referencia
ABI1 se usa exclusivamente para comparar. No se precarga en su memoria de entrada.

Las tres proyecciones deben entregar 6.144 resultados FP64 y 24.576 bytes FP32
exactos. Se comprueban todas las lecturas de pesos/activaciones, orden de
acumulación observable, escrituras, límites de 4 KiB y estados de terminación.
Las regiones comienzan a 4.032 bytes de su base para ejercitar particiones.

El productor reutiliza su memoria después del ACK de la cuantización, con una
identidad ajena y datos envenenados. La región ABI1 permanece retenida durante
O y hasta el ACK público: ambos hijos se liberan juntos. Se ejercitan pausas
AXI, identidades completas de 64 bits, acarreo de etiquetas y retención de
la terminación durante siete ciclos. Pesos y regiones deben permanecer
inmutables y disjuntos durante su propiedad.

## Evidencia disponible y criterio de aprobación

- Compilación sintáctica del banco real: PASS, con interfaces de IP sin comportamiento;
  no se ejecutó aritmética en esa comprobación. Archivo:
  `2a90f2a1baaa780af27342dab6fa93533f50752230786a1b3a19d920eabd39b7`.
- Propietario compuesto: 105 escenarios nativos, nueve mutantes de RTL rechazados
  y 57 controles negativos. Usó sustitutos de protocolo;
  [alcance y resultados](LLAMA32_ATTENTION_OUTPUT_PATH_RESULTS_R1.md).
- Esquema estático: 31 tipos de evento concordantes con el verificador.
  Sus 37 controles negativos **todavía requieren la traza real**.
- La aprobación exige XSim sin errores, comparación independiente completa,
  rechazo de los 37 controles, identidad de todas las fuentes compiladas,
  síntesis del top real sin cajas negras/latches y respaldo verificado.

La síntesis OOC usa `xcvu5p-flva2104-2-i`, reloj de 5 ns, dos CPU y límite
de 7.200 s. Informará recursos y timing; no califica colocación/ruteo ni E/S
externas. No valida DDR física, capa completa ni inferencia integral.

## Recuperación y procedencia

- Paquete: `20261013T224757Z-llama32-attention-output-path-real-vivado-r1`; SHA-256 `745707e87858de9abc715f98dac66938a249fbb6df5b24564034ae0f4f84e828`.
- ZIP de fuentes, verificado en D: y NFS: `ce368c2caef7878bade6179231b114eceae1f5a422f92654f9f4e7f0b3685f48`.
- Controlador: `e4aa11e1a01119d1ee3113691b4878d804983aa475efd92c12339d70636e03e7`.
- Coordinador: `build/cloud-setup/20261002-board/llama32-attention-output-path-real-vivado-r1-coordinator.json`.
- Auditoría de solo lectura: `attention-composition-r16-active-audit-20261013T224844Z.json`, SHA-256 `546dc8f7e4404529a1b6f4956f0fd0c929b8a9e3541f27d633089127a6932186`.
- La dependencia R2 adopta el registro corregido de O; no toma sus metadatos
  históricos erróneos como identidad de ejecución ni reinicia la herramienta.

El verificador exige que PASS ocurra exactamente en el ciclo de la última
liberación, tal como imprime el banco. El generador inicial del verificador
conserva su versión histórica; la fuente sellada y su hash son autoritativos.

Después de esta tanda: integrar la salida del bloque raíz real, después
residual, normalización y FFN. El cierre sobre DDR y la placa conserva las
dependencias de [la cola Vivado](board/VIVADO_QUEUE.md).
