# Composición de transporte post-O — R1

Estado 2026-10-15: banco nativo y revisión independiente aprobados.
`ntrpu_post_attention_path_r1.sv` conecta cargador AXI, propietario RAM y
escritor del par. HLS permanece como interfaz externa. **El sustituto HLS
reproduce salidas de referencia; esta tanda no acredita aritmética integrada.**

## Secuencia y propiedad

El controlador captura identidad de 64 bits y ambos destinos al admitir
el comando. El cargador captura las tres entradas. El llamador debe conservar
esas regiones inmutables hasta ACK final: `command_source_*` es una copia
del descriptor, no un handshake de liberación de otro productor.

La respuesta del cargador dispara el propietario; su respuesta dispara
el escritor. Los errores recorren la misma cadena para liberar recursos
propios sin iniciar cálculo o escrituras indebidas. Una cancelación breve
persiste entre etapas. La primera causa observada se conserva; la prioridad
simultánea es época, cancelación, cargador, propietario y escritor.

La respuesta retiene estado, identidad y conteos hasta ACK. La revocación
tardía invalida el descriptor sin cambiar el estado retenido. El cargador
sigue ocupado durante HLS y el propietario durante las escrituras AXI.
No se admite otro trabajo antes de liberar todos los hijos. Reset requiere
detener HLS y ambos pares AXI; el watchdog no autoriza reutilización temprana.

## Banco y evidencia

Tres semillas ejecutaron 48 trabajos cada una: **144 trabajos, 135 respuestas,
87 pares publicados y 1.069.056 bytes exactos**. Cada semilla incluye cuatro
casos normales y 22 escenarios seguidos de recuperación: identidad, límites,
RRESP/BRESP, error HLS, cancelación en cada etapa y en transiciones,
revocación tardía, timeout y reset coordinado.

El sustituto comprueba las tres entradas recibidas por AXI y entrega vectores
de referencia post-O. La memoria de destino se materializa desde AW/W
aceptados solo tras B correcto. El verificador reconstruye lecturas y bytes
confirmados, estados y orden; rechaza 39 corrupciones. Estabilidad AW/W,
forwarding RAM y revocación tardía son aserciones del banco congelado.
Los doce mutantes compilados del controlador fueron detectados. Cubren
captura de identidad/destinos, truncamiento, cancelación en admisión,
reutilización/ACK/publicación prematuros, conteos y diagnóstico HLS.
Cada mutante compiló y falló mediante una aserción explícita del banco.

`completion_read_beats` cuenta beats entregados por el lector, no todos
los R drenados. El caso RRESP fallido entrega tres y drena 65. El verificador
R2 diferencia ambos conteos, conservando el diagnóstico R1 sin cambiar RTL
ni banco. También se conservó el primer intento cuyo script omitía módulos;
el ejecutor R2 compila las seis fuentes necesarias.

- Banco: `20261015T075955Z-V7-post-attention-path-native-r2`.
- Archivo D:/NFS: `814ee8552169eb6b7e58125c2c9060a03296055461aafb576704e109cb36af7e`.
- Revisión R2: `af39634eccf5c7524209107a1ffa7f333567e240ed1265094ca4e36350d92292`.
- RTL: `7ef79cdd1a13815f49ba3d9d384627cc2ccb85f531195c1613ff21330e8cd8ac`.
- Mutantes: `20261015T082546Z-V7-post-attention-path-mutants-r1`.
- Archivo de mutantes D:/NFS: `abf6ad0f7935f3ab574ee2f00841ef41f5f4e8b51dacc475881dd14c0ebf4d06`.
- Revisión de mutantes: `2d5fa13c38cffc9849534a0d8db838f8f30c3e545107d6367615d935cf632985`.

## Unión HLS pendiente

El archivo terminal aprobado de post-O R3 confirma tres entradas `ap_memory`
de dirección 11 bits/datos 32, salida residual 11/32 y salida ABI1 9/64,
con CE/WE separados. `bind_ntrpu_post_attention_hls_ports_r1.py` inspecciona
los 26 puertos y rechaza cambios de ancho, dirección, puertos ausentes o
adicionales: 79 controles. Los índices 0, 1 y 2 corresponden
a residual conservado, O y peso de normalización.

El adaptador `fpga/rtl/ntrpu_post_attention_hls_ports_r1.sv` agrega cero
ciclos y conecta el reset síncrono activo alto del HLS. SHA-256:
`f9f53af6b343d33ab9ab81467fa167188950877e1060a94b6f12c0ab3586d303`.
Su módulo HLS de origen tiene SHA-256
`ac90a7877cd58d2ba630cbd3b1dc42fee190ae94d3bf704fc2cc074d5c40bce2`.
La inspección de interfaces no acredita simulación de la composición.

Las 26 transacciones reales tuvieron latencia máxima de 173.353 ciclos,
menor que el watchdog de 262.144. Esa duración completa acota cualquier
intervalo sin progreso para esos vectores; todavía debe observarse el
comportamiento del controlador durante la simulación integrada.

El [banco con RTL HLS real](LLAMA32_POST_ATTENTION_REAL_R1.md) pasó ocho
trabajos y confirmó 73.728 bytes, con cancelación/recuperación y catorce
controles. Su máximo intervalo sin progreso fue 165.137 ciclos, inferior
al watchdog. Ese resultado es independiente de esta aprobación nativa
de transporte; todavía falta unirlo a la raíz de atención y FFN.
