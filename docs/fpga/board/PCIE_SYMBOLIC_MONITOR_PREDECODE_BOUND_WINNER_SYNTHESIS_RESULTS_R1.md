# Límites B con winner: síntesis propia, tanda 995

## Resultado medido

`20261011T203802Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-winner-synth-r1`
terminó correctamente en **432,232 s**. La revisión R2 y la comparación
frente a RQ 984 aprobaron el archivo existente; no se repitió síntesis.
E y C conservan la referencia RQ para aislar el cambio B.

| Métrica | RQ 984 | B 995 | Cambio |
| --- | ---: | ---: | ---: |
| LUT | 92.913 | 93.179 | +266 |
| FF | 27.560 | 27.566 | +6 |
| CARRY8 | 568 | 573 | +5 |
| WNS estimado, ns | −2,768 | −2,758 | +0,010 |
| TNS estimado, ns | −1.472,519 | −1.470,562 | +1,957 |
| Endpoints setup incumplidos | 1.439 | 1.439 | 0 |
| WHS estimado, ns | −0,075 | −0,075 | 0 |
| Endpoints hold incumplidos | 2.068 | 2.068 | 0 |

Hay 71.169 endpoints y los doce controles de cobertura temporal dan cero
faltantes. No hay latches, BRAM, URAM ni DSP. Se mantienen dispositivo,
builds, reloj de 4 ns, retardos E/S cero y `RuntimeOptimized/rebuilt`.
La peor ruta ahora está en C: `allocated_tags_reg[51]` →
`read_range_high_reg[0]`, 28 niveles y dos CARRY8; 6,741 ns de datos,
con 1,679 ns de lógica y 5,062 ns de interconexión estimada.

## Corrección de revisión y siguiente decisión

El revisor R1 falló porque cinco referencias de su lista de avisos
esperaban el nombre anterior de B. El revisor R2 corrige únicamente
esas referencias y registra la derivación y el fallo original. Mantiene
los límites, cantidades y comprobaciones individuales de los avisos.
Los 209 avisos emitidos se revisaron; cuatro variantes incorrectas se
rechazaron. No se cambió RTL ni se sobrescribió evidencia previa.

Sigue el replay de la netlist propia y la comparación B→R por etapas.
La diferencia global de 0,010 ns no justifica promover el diseño;
siguen pendientes cierre físico y mediciones de rendimiento.

## Evidencia

Archivo local/NFS: 35.367.490 bytes, 186 miembros, SHA-256
`a349e259c02a5b507b4702ed376597a1212a8fd7095f6843d46673532a738a6a`.
Revisión R2: `e92e62d010f298a804130bb9cadb886af4fa41808a2b366186d51c5998efb2ff`.
Comparación: `0f5d5ada6aa13f67633917e3ee1f4ff24f8a4a190de9e9c9acfa1b4fdbb90062`.
Netlist funcional: `7dd15963670a55a1dec2eee537794b0fe2252a327045194ddc44b62d8751457c`.
DCP: `6f6b937d6aecde7923fc1943fbe0e0019196cfebfa8994897e1ada87ad259002`.

Certificados en `artifacts/ntrpu-board-vivado/`, carpeta de la tanda;
revisiones y recetas en `build/cloud-setup/20261002-board/`.
