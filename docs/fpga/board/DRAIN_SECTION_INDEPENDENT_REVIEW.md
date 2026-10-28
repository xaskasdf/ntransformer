# Revisión de composición y cruce de fallos

## Evidencia nativa revisada

La tanda `20261006T075415Z-V4-ffn-supervised-drain-sections-r5-frozen`
recuperó los cuatro perfiles del hardware original: reloj de dispositivo de
5/37 ns y contrato externo ausente/presente como fixture. El selector de
jerarquía parametrizada funciona; las capturas contienen 538–586 variables
y 4,56–5,80 MB por perfil.

`check_ntrpu_drain_sections.py` compara cada muestra AXI4 y cada transferencia
AXI-Lite contra VCD, verifica datos, strobes, atributos, colas causales,
identidades, respuestas retenidas, relojes detenidos y ausencia de reset
operativo. También exige desafíos ACK bajos nuevos para los siete slots.
Detectó 21 alteraciones deliberadas de la evidencia, incluyendo capturas
vacías/truncadas, muestras omitidas, datos/IDs incorrectos y marcadores falsos.

Por perfil se completaron 25 escrituras y 25 lecturas AXI4, con 1.061 beats
por dirección en la interfaz protegida. De ellas, 24 transacciones y 1.045
beats atravesaron el Clock Converter real; el resto fueron respuestas locales.
AXI-Lite completó seis transacciones por dirección, cinco reenviadas y una
local. Los perfiles con contrato externo registran además una AW tardía
pendiente, que invalida la prueba: no se declara drenada esa obligación.

Con contrato externo ausente nunca se afirmó `complete`. Con el fixture
presente se observaron dos pruebas completas de identidad nueva. La
revocación local llegó a gestión en 25.500 ps sin otro flanco host.

## Síntesis del proveedor R2

La tanda `20261006T080116Z-V4-ffn-supervised-local-reply-synth-r2b-frozen`
eliminó los 12 CDC-11 críticos por duplicación. Las seis fuentes registradas
reales conservan un único cruce XPM de tres etapas; la revocación local usa
otro. El informe contiene 41 CDC-3, diez CDC-9 y un CDC-15 pendiente dentro
del IP AMD. No se agregaron excepciones para ocultar los CDC-11.

Síntesis OOC: 10.809 LUT totales, 14.450 FF, WNS 1,523 ns y WHS 0,013 ns.
Faltan delays en 1.439 entradas y 1.720 salidas. Estas holguras no acreditan
timing físico ni frecuencia de una placa implementada.

## Criterios pendientes

La cola ejecuta ahora los cuatro perfiles con R2 y la respuesta R huérfana
que debe propagarse por la alarma compartida con el host detenido. Después,
`ntrpu_board_local_reply_negatives.tcl` prepara copias privadas con tres
defectos: omitir obligaciones locales, falsificar el tag de ACK y desconectar
la alarma compartida. Cada defecto debe fallar por su aserción específica;
el control positivo debe terminar normalmente.

Los resultados independientes son addenda con hashes del verificador, logs y
VCD; las tandas originales y sus certificados permanecen intactos. Siguen
abiertos los proveedores externos reales, integración de reset del endpoint,
CDC-15 y la calificación física DDR/PCIe de la tarjeta de 512 GiB.
