# Habilitaciones de escritura por fila en E/D — R1

## Motivo y cambio

El [ruteo Quick](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_DIAGNOSTIC_R1.md)
midió 17,145 ns de interconexión entre la selección de admisión registrada
en E y el enable de una fila D, con fanouts 54 y 51. La síntesis RQ 984
mantiene márgenes estimados positivos en E→D; eso no elimina el problema
físico observado en el antecesor. No se atribuye al candidato una mejora
de colocación antes de medirla.

El borrador parte del par E/D original de RQ. E captura, junto con su
registro existente, 256 habilitaciones de descripción y 256 de atributos.
Cada máscara incluye valid, observing y las operaciones de esa fila.
D usa esas señales para escribir y conserva el orden RC, RQ0, RQ1.
El refinamiento de fuentes R2 elimina el mux redundante de observing en
los datos de tabla; salidas y contexto conservan sus condiciones originales.
No añade etapas ni cambia reset frío, lecturas o capacidad. Los 512 bits
declarados adicionales no son un conteo de FF sintetizados.

## Prueba encolada

El banco conecta ambos pares E/D reales a la misma entrada C pública.
Compara antes y después de cada flanco las 256 filas completas: **14.336
bits de descripción y atributos**, 51 del contexto retenido, 2.498 de
salidas D y 4.554 de salidas E originales. Una ecuación independiente
sobre E de referencia verifica los 512 bits nuevos. También comprueba
estabilidad antes del flanco; no fuerza estado interno.

Las **40.958 filas** incluyen 16 dirigidas, 16.384 combinaciones binarias
para todas las filas, 16.384 casos de seis controles en cuatro estados,
7.148 inyecciones X/Z en cada bit C, 1.024 registros aleatorios y dos
ciclos de drenaje. Hay oráculos explícitos de admisión, commit y prioridad.
Seis fuentes defectuosas prueban máscaras, valid, observing, colisión y
reset mediante primeros testigos predeterminados. El revisor rechaza
39 corrupciones de evidencia.

## Dependencias y criterios

Se ejecutará después de los tres [diagnósticos E/B/C](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMPONENT_STAGE_QUEUE_R1.md),
con snapshot nuevo, dos hilos y límite de 1.800 s. Un fallo detiene la
cola y se respalda; no se reinicia automáticamente. R17 conserva fuentes,
orquestación y snapshots pendientes en D: y NFS.

La auditoría estática y el modelo independiente de 12.288 transiciones
son comprobaciones previas. Se exige PASS RTL de cobertura completa y
mutantes antes de integrar el monitor. Después serán necesarias síntesis,
replay de la netlist propia y comparación física equivalente. Este banco
no acredita timing, throughput, PCIe nativo ni placa.
