# Monitor R1: oráculo y controles de quiet pendientes

## Prototipo y dependencia

`fpga/rtl/ntrpu_pcie_symbolic_monitor_r1.sv` compone C3/E2/D2/A1/B1/R1.
Es un prototipo sin aprobación de integración. Ejecutar su prueba después
de revisar y respaldar la reproducción de la netlist propia de R.
Mantener los WNS negativos de C/E/D y R como limitaciones abiertas.

## Adaptación del banco

Derivar un banco de `tb_ntrpu_pcie_symbolic_retirement_r1.sv` mediante
reemplazos literales registrados. Instanciar el monitor como `ced`, quitar
las instancias A/B/R externas y observar las instancias internas del
contenedor. Conservar todas las comparaciones originales de estados,
tablas, late faults y origen. La comparación C/E/D debe observar
`d_control`/`out_record`, no las nuevas salidas públicas ya retiradas.

Para quiet, usar exclusivamente R7/S1 y su historia independiente:

- Antes del flanco, el estado publicado esperado es el guardado después
  del flanco anterior; los cinco registros históricos representan C..B.
- Después del flanco, antes de desplazar la historia, R espera la posición
  4 antigua. Los pendientes son el registro recién capturado más las
  posiciones 0..3 antiguas. El origen de B que acaba de retirarse se excluye.
- El control vivo esperado usa counts/parciales/held de R7 y las entradas
  actuales. Los fallos tempranos proceden de S1, ya contrastado con R7/modelo.
- Requester y atributos esperados proceden del arming de R7. Comprobar
  quiet por igualdad de valores conocidos, incluidos frío, ready, perfil,
  link, identidad, fallos, bitmap, counts, bytes y parciales.

Contabilizar muestras antes/después del flanco, los cinco vaciados y los
testigos de vacío. No construir expectativas desde salidas del candidato.

## Fixture dirigida y fallos introducidos

Incluir arming demorado, escritura posted completa que deje counts a cero,
lectura completada con fallo tardío aislado, pérdida/restauración de perfil,
frío ocupado y último retiro. Añadir pruebas entre flancos que cambien y
restauren cada guarda, incluyendo X/Z, sin cruzar un flanco ni alterar los
vectores originales. Acreditar restauración y revocación inmediata.

Detectar controles reales que omitan cada pendiente C/E/D/A/B, que cuenten
el efecto ya retirado de R un ciclo extra, que omitan fallo temprano o
tardío y que ignoren frío, observación, requester, atributos o ready.
Para ignorar RQ/RC inmediato hay que quitar también su contribución al
control vivo: quitar solo la guarda explícita quedaría enmascarado por
`c_control_pending`. No exigir detección de mutantes funcionalmente
equivalentes por redundancia de guardas.

Ejecutar primero los testigos dirigidos y sus controles, después los
bancos originales con X/Z y vaciados. Cada control debe fallar por la
comparación de quiet/pending prevista. La aprobación funcional no
acredita colas ocultas PCIe, recepción posted, cierre físico ni placa.
