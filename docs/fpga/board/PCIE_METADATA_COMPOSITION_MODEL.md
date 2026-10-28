# Composición de efectos de metadatos: estudio binario R1

Ejecutado el 2026-10-09 UTC con
`tools/check_ntrpu_pcie_metadata_composition_r1.py`. Resultado:
`build/cloud-setup/20261002-board/pcie-inventory-metadata-composition-model-r1.json`.
Es el primer estudio parcial del [siguiente corte](PCIE_INVENTORY_REGISTERED_SUM_FOLLOWUP.md).

## Representación y referencia

Se estudian exclusivamente `remaining[12:0]` y `address[11:0]` de una fila.
La entrada ya es un efecto seleccionado: consumo normal, reemplazo RQ o
identidad. El resumen conserva el último reemplazo, presencia de consumo,
primera cantidad y suma de cantidades posteriores. Un reemplazo posterior
descarta los efectos previos sobre esa fila; no elimina obligaciones ni el
payload RC antiguo del monitor completo.

La referencia recorre los eventos uno por uno. Calcula `available` con
15 bits, lo compara como bits sin signo con `remaining`, y aplica truncamiento
de 13/12 bits. La alternativa compone resúmenes y aplica una expresión cerrada
desde la semilla canónica. Conserva el caso especial de primer count cero
con dirección desalineada. Los anchos y asignaciones proceden de R7; no se
usa el byte count recibido para sustituir los metadatos esperados.

## Evidencia

Pasan **131.600 casos**, con cadenas de 1–16 efectos y **518.208 comprobaciones
de agrupación**. 99.159 casos contienen reemplazos. Se incluyen semillas
en límites, direcciones que envuelven, counts cero y cantidades de 11 bits.
Cinco controles rechazan omitir el caso cero/desalineado, truncar la suma
a 11 bits, conmutar consumo y reemplazo, conservar la semilla anterior a
un reemplazo y omitir el wrap de dirección.

Ejecución observada: 14,345 s. Reproducir con una ruta de salida nueva:

```powershell
python -X utf8 tools/check_ntrpu_pcie_metadata_composition_r1.py <resultado-nuevo.json>
```

## Límites y siguiente puerta

La suma usa enteros Python sin límite: todavía no se elige una ventana de
hardware ni su ancho. El estudio no cubre TC/atributos, selección de eventos,
pertenencia/versiones de tags, payload retenido, X/Z, reset/vacío/latencia de
etapas, RTL o timing. Las cadenas son finitas y muestreadas; no constituyen
una prueba universal de asociatividad del circuito.

Extender la representación y su referencia antes de conectarla al banco de
metadatos: obtener los efectos desde tráfico crudo, resolver generaciones y
contexto retenido, mantener los casos posteriores a fallo y comprobar X/Z.
Después fijar etapas, forwarding, anchos y aceptación sostenida RQ2/RC4.
El monitor segmentado completo sigue sin implementar ni calificar.
