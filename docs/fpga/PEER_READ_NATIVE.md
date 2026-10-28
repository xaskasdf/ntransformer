# Lectura peer con HBM nativa

## Objetivo y alcance

Validar el candidato de [corte registrado](PEER_READ_CUT.md) con el modelo HBM
de AMD. El preparador toma una copia congelada de la composición que pasó
ruteo; comprueba sus hashes y conserva los 13 archivos RTL de esa composición.
Las configuraciones `CONFIG.*` de las tres IP de lectura deben coincidir.
Las rutas locales de los archivos de IP no forman parte de esa comparación.

El banco escribe los datos mediante el receptor de escritura existente y lee
por `ntrpu_peer_read_pair`: ledger, dos frames por canal, conversión de ancho,
protocolo, CDC y cola registrada de retorno. El banco verifica los bytes de la
salida; no fuerza señales internas ni modifica la memoria del modelo.

## Casos y aceptación

- Precargar 20 transferencias, 37.376 bytes, en dos canales. Publicar la región
  de prueba después de recibir todas las completions de escritura.
- Mantener cuatro contextos simultáneos, aplicar contrapresión a la salida y
  conservar la reserva hasta completion local, notificación DMA y ACK remoto.
- Reutilizar contextos con longitudes de 64, 512, 576 y 4.096 bytes; rechazar
  el replay de una identidad retirada.
- Detener ambos canales con comandos pendientes y conservar dos contextos en
  cuarentena después de aceptar los resultados de fallo.
- Auditar cada escritura, fragmento AXI nativo y byte público; reproducir las
  20 transferencias exitosas con el modelo de transporte independiente.
- Comprobar conservación, orden, ocupación y `pending` de la cola de retorno.
  Los controles negativos deben detectar evidencia ausente o alterada.

La precarga es un fixture de prueba, no un gestor completo de publicación.
DMA y ACK son estímulos del banco: no acreditan un enlace PCIe ni otra tarjeta.
Este perfil no reproduce los 21 escenarios de fallos del banco conductual.
La simulación nativa es RTL, sin SDF; el netlist ruteado tiene evidencia separada.
No se interpreta el tráfico corto y pautado como medida de ancho de banda máximo.
Las direcciones de origen son `0x0` y `0x10000`, como en el oráculo previo;
dos puertos AXI no demuestran aquí acceso paralelo a dos stacks HBM locales.

## Ejecución y respaldo

`tools/prepare_ntrpu_peer_native.py` genera el banco en un directorio nuevo a
partir del candidato revisado. `tools/run_ntrpu_peer_native.py` separa tres
etapas: `preflight`, `compile` y `traffic`. La primera usa almacenamiento ligero;
la segunda copia y verifica una caché HBM revisada; la tercera ejecuta una copia
del snapshot compilado. Cada etapa conserva fuentes, resultados y manifiestos.

`tools/check_ntrpu_peer_native.py <salida>` revisa la evidencia antes de aceptar
una tanda. Los artefactos se guardan por contenido y se restauran para repetir
la revisión. También se conserva el snapshot ejecutable: 15 archivos,
510.901.480 bytes antes de compresión, identificados por SHA-256. El manifiesto
de permisos POSIX permite restaurar su ejecución en Linux con Vivado 2026.1.

```bash
python3 -B tools/prepare_ntrpu_peer_native.py SELECTED SELECTED/REVIEW.json PRIVATE \
  --ip-origin ORIGINAL_PHYSICAL
python3 -B PRIVATE/tools/run_ntrpu_peer_native.py PREFLIGHT --stage preflight
python3 -B PRIVATE/tools/check_ntrpu_peer_native.py PREFLIGHT > PREFLIGHT_REVIEW.json
python3 -B PRIVATE/tools/run_ntrpu_peer_native.py COMPILE --stage compile --threads 8 \
  --cache HBM_CACHE --cache-review HBM_CACHE_REVIEW.json \
  --preflight PREFLIGHT --preflight-review PREFLIGHT_REVIEW.json
python3 -B PRIVATE/tools/check_ntrpu_peer_native.py COMPILE > COMPILE_REVIEW.json
python3 -B PRIVATE/tools/run_ntrpu_peer_native.py TRAFFIC --stage traffic \
  --prior COMPILE --prior-review COMPILE_REVIEW.json
python3 -B PRIVATE/tools/check_ntrpu_peer_native.py TRAFFIC
```

Activar antes el entorno Vivado 2026.1. Cada salida debe ser un directorio nuevo.
El lanzador de tandas limita además la afinidad de CPU; los scripts usan prioridad
baja y un máximo de 1.800 segundos por proceso de herramienta.

## Estado

La prevalidación con almacenamiento ligero pasó en 70,907 s: 20 escrituras y
20 resultados correctos, 37.376 bytes comparados, dos contextos retenidos tras
la parada y 15 controles negativos rechazados. La cola registró 584 entradas y
584 salidas, 73 muestras llena y ningún dato residual.

La compilación nativa pasó en 500,257 s, sin contar la copia y verificación de
la caché. El snapshot está respaldado y su restauración pasó la revisión.
La primera ejecución no arrancó porque el lanzador perdió el permiso ejecutable
al copiar `xsimk`. Se archivó como fallo de lanzamiento y se corrigió la copia
con `copy2`, comprobando también los modos POSIX. Solo cambió el lanzador;
fuentes simuladas, configuración y bytes del snapshot permanecen idénticos.

**La ejecución corregida pasó y fue revisada independientemente.** Duró
1.033,597 s; terminó a los 61,695 µs simulados. Ambos controladores indicaron
inicialización completa antes del arranque a los 48,415 µs. Pasaron las
20 transferencias, los 37.376 bytes escritos/leídos/entregados, la confirmación
remota, la reutilización y la retención de dos contextos tras la parada.

El auditor observó 51.296 muestras de la cola: 584 inserciones y 584 retiradas,
33 muestras retenidas, 15 llena, ocupación máxima de dos y cero palabras
residuales. Rechazó nueve alteraciones de evidencia de memoria/peer y seis de la
cola. Las 20 transferencias se reprodujeron en el modelo de transporte
independiente. Las señales térmicas permanecieron `X`; no se calificó esa función.

| Evidencia | SHA-256 |
| --- | --- |
| Checkpoint físico conservado | `281a71072ce22ed2ec52460013c111f2b689fef8fa0d20730463ea2909b85dae` |
| Resultado de compilación | `2f4a0c8e0e4387b3521abbd7c04617667824c08c4ca31b3106ff328ea2c3b011` |
| Manifiesto del snapshot | `ddf85a1822e03de69e428624bc6d24499e49be4d61b3dc06c5b3f97c0ae3cea5` |
| Resultado nativo aceptado | `90a86ed800fcc704ba2b28ef1b3d1d5587a3253d4506185d5f2982d10a281c55` |

Tandas y entrega local:
`D:\AI\artifacts\ntrpu-peer-native\20261007T233159389882Z`.
`preflight2`, `compile` y `traffic2` son las etapas aceptadas; `traffic` conserva
el fallo de lanzamiento. Sus respaldos en `ntrpu-content-store` se restauraron
y verificaron. La compilación añade unos 292 MB comprimidos al almacén; la tanda
nativa aceptada añade unos 526 KB. No se conservan ZIP sueltos.

El timing conservado sigue siendo interno y con margen estrecho. La posterior
[admisión compartida](PEER_READ_SHARED.md) ya pasó pruebas funcionales, pero debe
repetir timing y HBM nativa para su nueva composición. Quedan el gestor de regiones,
multispan, rendimiento sostenido con localidad HBM, integración PCIe y cierre
físico de placa. Esta prueba no hereda los 14,711 GB/s medidos en F1 aislado.
