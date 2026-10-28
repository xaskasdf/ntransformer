# Coste medido de preparación de las simulaciones

## Medidas de tandas revisadas

Se leyeron las consolas originales de los archivos certificados, sin
ejecutar otra tanda. Ambos bancos contienen 22 casos completos.

| Medida | RTL, tanda 961 | Netlist, tanda 963 |
| --- | ---: | ---: |
| Duración total registrada | 777,337 s | 1.503,802 s |
| Compilación, suma reportada | 63 s | 61 s |
| Elaboración, suma reportada | 218 s | 416 s |
| Compilación + elaboración después del primer caso | 268 s | 335 s |
| Esa preparación posterior respecto al total | 34,477 % | 22,277 % |

Vivado reporta estos pasos en segundos enteros. El tiempo de
`launch_simulation` ya los incluye: no se suma otra vez. La netlist ya
comparte proyecto y aprovecha compilación incremental; su primera
elaboración duró 133 s y las siguientes aproximadamente 13–14 s.

## Mejora del flujo por evaluar

Los bancos y controles de entrada se seleccionan mediante plusargs.
Una futura revisión puede estudiar compilar/elaborar una vez y arrancar
un proceso de simulación nuevo para cada caso. Los 268/335 s son coste
observado, **no ahorro demostrado**. La herramienta puede necesitar parte
de esa preparación aun reutilizando la instantánea compilada.

Antes de adoptar el cambio hay que verificar identidad de fuentes y
opciones, bibliotecas y snapshot; conservar comandos, logs y argumentos
por caso; y demostrar reinicialización completa. Deben seguir ejecutándose
los 16 bancos, los seis controles negativos y todas las comprobaciones
anteriores/posteriores al flanco, quiet, drenaje y primeras discrepancias.
La revisión independiente debe enlazar cada resultado al binario realmente
ejecutado. No se modifica la cola congelada en curso.

## Evidencia

`profile_pcie_symbolic_monitor_predecode_simulation_cost_r1.py`, en
`build/cloud-setup/20261002-board/`, verifica hashes de archivos, revisiones,
evidencias, consolas e invocaciones. Su resultado registra los tiempos de
cada caso, incluidos los controles negativos.

Perfil SHA-256:
`fa64012f63f65a34edeee708d2b340f6a8f2e316efcc1e69829d69a79a70ab2a`.
No se cambió el banco, no se lanzó Vivado y no se atribuye rendimiento
de hardware a estas duraciones.
