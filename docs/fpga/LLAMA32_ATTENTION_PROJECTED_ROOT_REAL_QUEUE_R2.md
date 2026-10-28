# Raíz de atención → cuantización → O: banco real en cola

**Actualización 2026-10-15:** R2 terminó con un diagnóstico de alineación
de etiquetas en el banco. Su evidencia está respaldada. La revisión R3,
con banco R2 corregido y el mismo RTL, está ejecutándose. Véanse
[causa, resultados de las dependencias y cola vigente](LLAMA32_PROJECTED_ROOT_TAG_RECOVERY_R3.md).
El resto de este documento describe el paquete R2 original.

Estado al 2026-10-14: **paquete R2 sellado y encolado, sin ejecución
aritmética integrada todavía**. Espera la aprobación y el respaldo de
cuantización → O R3, que a su vez espera la regresión O con el watchdog
corregido. El turno Vivado continúa siendo secuencial.

## Diseño y fuentes

El top es `ntrpu_attention_projected_root_r1`, con los módulos reales de
caché K/V, raíz de 32 cabezas, cuantizador FP32→ABI1, escritor y proyección
O mediante dos MAC. Comparte AXI según el propietario y bloquea operaciones
de mantenimiento hasta liberar la respuesta pública. Conserva secuencia
y época de 64 bits durante los traspasos.

El paquete contiene 88 fuentes compilables: 34 RTL propios, el banco y
53 Verilog generados —27 de atención, 18 de cuantización y 8 del MAC—,
más los scripts de sus IP. Incluye el controlador
`registered_watchdog_r3/ntrpu_attention_projection_control_r1.sv` y las
entradas `wire` explícitas. El lector AXI compartido usa la variante
`vivado_explicit_nets_r1`, cuyo cuerpo y puertos normalizados coinciden con
el lector empleado por O. No se compilan sustitutos de los núcleos HLS.

La compilación previa `20261014T042022Z-V7-attention-projected-root-bank-compile-r2`
pasó con las tres interfaces HLS vacías. Comprueba conectividad del banco,
no aritmética. La prueba nativa anterior del propietario conserva sus
93 trabajos, diez defectos RTL detectados y 78 controles de traza.

## Banco y criterios numéricos

Cada trabajo enlaza una caché con capacidad de cuatro tokens y escribe
los cuatro K/V de la capa cero mediante AXI. Las memorias K/V y ABI1
empiezan sin inicializar; los datos se materializan al aceptar B. Las
referencias precalculadas se usan para comparar, no para alimentar las
lecturas de los consumidores.

| Trabajo | Entrada | Resultado requerido |
| --- | --- | --- |
| 1 | Prefijo de un token | 32 cabezas y salida O completa |
| 2 | Cancelación tras ocho beats Q | Estado 7; ninguna cabeza, cuantización ni O |
| 3 | Prefijo de cuatro tokens | Recuperación con otra salida O completa |

Totales esperados: 12 appends, 264 beats Q, 64 cabezas, 4.096 valores FP32
de atención y 4.096 resultados O FP64; 16.384 bytes FP32 finales. K/V debe
mostrar 204 AW/B, 768 W, 160 AR y 1.280 R; ABI1, 128 AW/W/B. El verificador
comprueba bytes, direcciones, divisiones de página, identidades, orden de
confirmación y retención hasta ACK. Reconoce 47 tipos de eventos y
ejecutará 67 controles de corrupción sobre la traza completa.

## Ejecución y respaldo

El Tcl sintetiza primero el top real para VU5P, 5 ns, y rechaza cualquier
lazo combinacional antes del banco largo. Después ejecuta XSim con ambos
fixtures congelados. La revisión exige salida completa, fuentes realmente
compiladas, cero cajas negras/latches y cero endpoints internos sin
restricciones; el timing OOC no equivale a ruteo aprobado. El límite es de
dos CPU y 7.200 segundos. La evidencia se transfiere comprimida y el turno
se libera solo tras revisión y respaldo verificado.

- Coordinador: `llama32-attention-projected-root-real-vivado-r2-coordinator.json`.
- Paquete SHA-256: `98ecb2fc6244e29e5db5bcf0f778cbb5d668163fe027ce50c17217b99e05fd7d`.
- ZIP de fuentes SHA-256: `627455c91709178cb0dc21979d45366aeb1784d1ae061732282b1e48fb0555ee`.
- Compilación previa, archivo SHA-256: `4228803f40546c042d76b561cb0637da35db8f71d101e64e1d6c4209c87d1f00`.
- Fuentes y compilación verificadas en `D:/AI/ntransformer/artifacts/` y NFS.

R1 se conserva sin lanzar: la revisión previa detectó un nombre de
checkpoint heredado en el recolector. R2 corrige esa ruta y fue revisada
antes de encolarla; no cambió el RTL ni el banco.

DDR física, PCIe, ruteo integrado, residual/normalización/FFN e inferencia
completa siguen fuera del alcance de esta tanda y pendientes en la cola
general. Véase también la [corrección del watchdog](LLAMA32_ATTENTION_O_WATCHDOG_RECOVERY_R3.md).
