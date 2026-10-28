# Modelo binario del intervalo de payload RC

## Resultado y alcance

El modelo local comparó exhaustivamente **67.108.864 combinaciones**:
contador `seen` de 0 a 2047, desplazamiento de dirección de 0 a 3 y
`delivered` de 0 a 8191. Son **268.435.456 predicados de byte**. Ambas
formulaciones produjeron la misma máscara de cuatro enables en todos los
casos, incluidos valores que representan tráfico malformado.

Esto comprueba una identidad de enteros binarios. No verifica todavía el
ancho contextual de expresiones SystemVerilog, X/Z, selección SOP/EOP,
integración RTL, síntesis ni mejora de timing. El diseño activo no cambia.

## Factorización comprobada

La expresión original habilita el byte `j` cuando
`(seen−3)*4+j ≥ offset` y `(seen−3)*4+j < offset+delivered`, dentro del
guard existente para `seen ≥ 3` y contexto normal.

Definiendo `word=seen−3`, `end=offset+delivered`, `q=end//4` y `r=end%4`:

- Máscara inferior: `(15 << offset) & 15` en la primera palabra; 15 después.
- Máscara superior: 15 antes de `q`, `(1 << r)−1` en `q`, cero después.
- Enable: AND de ambas máscaras; cero si `seen < 3`.

El límite máximo es **8194**: un almacenamiento de trece bits truncaría
entradas posibles. El banco conserva todo el dominio codificado y no lo
recorta a tamaños de payload considerados válidos.

## Controles y evidencia

Seis alteraciones fueron rechazadas: truncar el límite a trece bits,
omitir la alineación inicial, incluir el extremo superior, tratar resto
cero como palabra completa, adelantar la última palabra y limitar
`delivered` artificialmente a 4096.

El script `prove_pcie_symbolic_monitor_predecode_payload_interval_r1.py`
en `build/cloud-setup/20261002-board/` enlaza por SHA-256 el RTL original
y el candidato por prefijo. El resultado guarda histograma completo,
hash del flujo de máscaras y los seis contraejemplos. Duración: **1,341 s**.
Prueba SHA-256:
`b3ca9d8327928f54d6b52d04379a66442f7e17fa11dddeb7f5eb9705dee64000`.

## Uso previsto

La [hipótesis de límite precalculado](PCIE_SYMBOLIC_MONITOR_PREDECODE_LATE_TIMING_NOTES_R1.md)
debe evaluarse después de las medidas del contexto por prefijo. Separar
la suma de la selección y factorizar enables son cambios distintos: una
prueba de uno no valida el otro. Cualquier candidato posterior deberá
preservar las condiciones procedurales, pasar RTL de cuatro estados e
integración completa y medir su propia netlist antes de promoverse.
