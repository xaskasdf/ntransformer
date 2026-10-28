# Predecodificación del bitmap RQ: experimento enfocado R1

## Motivo y cambio

La síntesis 979 redujo área, pero su ruta C→C llega a la escritura del
bitmap después de seleccionar un tag de admisión. Este candidato compara
los cuatro tags RQ originales con las 256 filas antes de la decisión de
admisión; al final aplica los bits de admisión. Conserva prioridad RC→RQ,
latencia, contadores, metadatos y selector RC. Es un módulo C separado:
el padre y las referencias existentes permanecen intactos. Todavía no
hay mediciones de síntesis ni timing de este cambio.

## Evidencia local

El modelo finito R2 aprobó 3.483.648 comparaciones: 4.096 combinaciones
de controles de dos bits en 0/1/X/Z, 21 mapas SOP distintos, máscaras de
lectura/admisión, igualdad de tags 0/1/X y estado previo del bitmap.
Detectó cinco alteraciones del modelo. El diagnóstico R1 se conserva:
una comprobación textual confundía los dos bucles SOP, RC y RQ, y falló
antes de enumerar casos. El modelo no prueba de nuevo la función de
admisión ni sustituye la simulación de SystemVerilog.

## Tanda prevista

Después del replay 980 y de la revisión comparativa de sus DCP:

- Ejecutar el C real y su padre sin inyectar estado interno.
- Comparar 42.910 filas antes/después del flanco: registro C de 3.574 bits,
  `control_pending` y bitmap combinacional de 256 bits.
- Cubrir 8.192 escenarios SOP/tag con X/Z, 256 combinaciones de tipos,
  64 handshakes y las 8.850 filas del suplemento RQ existente, con
  capacidad y liberación RC seguida de reutilización RQ.
- Compilar cuatro fuentes defectuosas por separado: omitir lane 3,
  rotar tags, aceptar igualdad desconocida y omitir admisión.
- Exigir sus divergencias exactas y rechazar 16 corrupciones de evidencia.

El banco usa únicamente entradas públicas; conserva el tráfico tras
fallos. El límite es 1.800 s y dos hilos. Cada desenlace debe archivarse
en D: y NFS, incluso un fallo. El PASS habilita la regresión completa del
monitor; no califica aún integración, netlist, timing ni hardware.

## Archivos y respaldo

El prefijo de ejecución es `pcie-symbolic-monitor-predecode-rq-bitmap-focus-r1`
en `build/cloud-setup/20261002-board/`. El manifiesto `rq-bitmap-r1-sources`
registra las cuatro ediciones exactas y las fuentes mutantes. La cápsula
de trabajo R8 añade esos bancos y futuros snapshots RQ a la cobertura
de R7, conservando el checkpoint 945 y los paquetes históricos.

La preparación R2 corrige, antes de congelar o ejecutar el banco, un
marcador de exportación heredado en el recolector. El generador registra
el hash anterior y la edición; la primera cápsula R8 conserva el borrador.
Los nombres de ejecución y owner siguen siendo R1. Se verificaron las
recetas exactas, el enlace al owner y la coincidencia del marcador Tcl.
