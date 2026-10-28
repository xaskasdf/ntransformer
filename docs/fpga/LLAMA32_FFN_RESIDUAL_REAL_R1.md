# FFN completa y residual con aritmética real

Actualización: 2026-10-16, 01:20 UTC. **Simulación aritmética integrada
aprobada; síntesis medida con timing pendiente**. La revisión R2 utiliza
la tanda original, sin repetir Vivado ni cambiar RTL o estímulos.
La [cualificación nativa de control](LLAMA32_FFN_RESIDUAL_CONTROL_R1.md)
es una evidencia distinta.

## Resultado ejecutado

Tanda `20261015T211706Z-V7-ffn-residual-real-vivado-r1`: XSim verificó
un caso completo con 56.770.560 bytes leídos, 163.840 escritos y un vector
final de 8.192 bytes exactos. Duró 20.345.899 ciclos simulados; el HLS
residual tomó 4.110 ciclos. No son medidas de una tarjeta física.

| Síntesis OOC, reloj de 5 ns | Resultado |
| --- | ---: |
| LUT / registros | 46.738 / 42.682 |
| BRAM de 36 Kib equivalentes / DSP | 48,5 / 93 |
| WNS setup / WHS hold | −1,770 ns / +0,039 ns |

**No cierra timing a 200 MHz**, incluso antes de implementar/rutear.
No hay latches, cajas negras ni bucles combinacionales detectados.

El revisor R1 exigía que `HLS_DONE` ocurriera después del último
`HLS_WRITE`; ambos aparecen en el ciclo 20.341.160. El propietario RTL
cuenta la escritura de ese mismo flanco antes de comprobar la extensión.
El [revisor R2](../../tools/check_ntrpu_ffn_residual_real_r2.py) acepta esa
coincidencia y mantiene orden, 2.048 valores exactos y rechazo de fin
prematuro. Pasaron **64 controles negativos** sobre la traza real.
El diagnóstico R1 queda conservado; corregir el revisor no elimina el
déficit temporal de la síntesis.

Archivo completo: SHA-256
`e4d909de667b06c66d61119ff1cd503c50ed169aa994b6b42a9f55fd80a66b79`.
Revisión R2: `50937f4bd765423e48b74281fd0c4eed89dc68f44eb147ee6bb06e9b8e09baa3`.
Suplemento: `1d31d1eaa52f53e70ed8927fa78aa71bc138b767bc87b4642cc3e060934e2605`.
La tanda completa y las fuentes se restauraron en D: (**334 + 137 archivos**),
además del respaldo NFS. La cola continúa con post-O → atención/FFN →
BRAM → actividad DDR, conservando todos los trabajos siguientes.

## Cadena y memoria

El top `ntrpu_llama32_ffn_residual_r3` ejecuta la FFN de 2048/8192 con
las matrices Q8 originales del GGUF y añade el residual mediante los
nueve Verilog HLS originales. Conserva la entrada ABI1 exacta post-O,
muestra 1/token 0. El productor post-O aún se representa por su fixture
verificado; no se ejecuta una inferencia completa.

El [banco completo](../../fpga/tb/tb_ntrpu_llama32_ffn_residual_real_r1.sv)
usa diez regiones: entrada, tres matrices, cuatro resultados FFN,
residual retenido y salida final. Sólo la recepción de B con éxito
materializa los datos W. El residual lee esos bytes FFN y su entrada
retenida. Los resultados esperados permanecen en arrays separados,
sin precargarse como entradas del DUT.

| Contrato esperado | Bytes |
| --- | ---: |
| Matrices Q8 originales | 53.477.376 |
| Lecturas de la FFN | 56.754.176 |
| Lecturas adicionales del residual | 16.384 |
| Lecturas totales | 56.770.560 |
| Escrituras FFN / total con residual | 155.648 / 163.840 |
| Vector final FP32 | 8.192 |

Son volúmenes exigidos por el banco, no medidas de hardware. Los
contadores separan gate, up, SwiGLU, down y residual; registran el tiempo
del HLS residual, primer AR a último B, respuesta pública y ACK.

## Preparación conservada y comprobaciones

El [revisor independiente](../../tools/check_ntrpu_ffn_residual_real_r1.py)
verifica el plan completo AR, checksums de lectura, datos/máscaras W,
confirmaciones B, lecturas de datos producidos, las 2.048 sumas FP32,
identidad pública de 64 bits, seis ciclos de backpressure final y
liberación simultánea de los tres niveles de propiedad. La suma usa
el verificador entero independiente de redondeo FP32.

El self-test del revisor rechazó **60 alteraciones** de una traza
sintética. Esto comprueba el revisor, no ejecuta la aritmética RTL.
La comprobación `xvlog/xelab` de **38 fuentes** analizó el banco y
elaboró el top con dos sustitutos de interfaz MAC/SwiGLU; tampoco
ejecutó aritmética. El paquete de simulación completa excluye esos
sustitutos y contiene **68 fuentes efectivas**, más los once archivos
de simulación IP que Vivado debe generar y cuya identidad se revisa.

## Cola, límites y evidencia original de preparación

Registro histórico del 15 de octubre: coordinador
`queue_ffn_residual_real_vivado_r1.py`, PID local 19176,
estado `waiting_for_ffn_residual_handoff_review_and_backup` a las 18:35.
Espera el traspaso R1 y también verifica explícitamente la aprobación
y respaldo de la FFN R4. Después ejecuta síntesis OOC a 5 ns y el caso
completo XSim, con dos CPU y límite de ocho horas. Conserva diagnósticos
y respalda cada resultado antes de liberar la cola.

| Artefacto | SHA-256 |
| --- | --- |
| Fuente sellada, 137 miembros | `163fb91e9575741081e4e772c0fcb4b74b79451faba914b883ea4afcaf910dc4` |
| Configuración del paquete | `cf0cfa9a8214604d8fd22baadb1e4977e59eb2ccaf803dfc92bc50f6c3e3196c` |
| Banco real R1 | `fe9c2cf3138d2ef0b87146e650361be555da030251c1b785579ba7b3c1c29100` |
| Fixture adicional | `b14e8731ce63420cbd1d775f241b8f621e491caf189d8deab2a803136f7a49e0` |
| Archivo de compilación previa | `5ed3b1ed5ddeaf2ab484fca66e766202ca5931deec5717fbbeba9b4fc243b6b9` |
| Revisión de compilación | `67c169fd874c43fcf7a19305d033358bb248e360a406591310432382b658f392` |
| Self-test del revisor R2 | `a0d079c7ff38b2d1a8fb09fed1d5ff498590a00864a2dd6e6b78cc701b47a7e0` |

Paquete local: `D:/AI/ntransformer/artifacts/ntrpu-model-validation/20261015T183359Z-ffn-residual-real-vivado-r1/`.
La copia NFS y cada miembro se verificaron por hash. Las correcciones
R1–R3 del generador de preflight y R2 del self-test se conservaron;
no alteraron las fuentes compiladas de etapas anteriores.

Durante la preparación, el auditor R40 confirmó siete coordinadores
vivos y FFN R4 en XSim. Siguen abiertos productor post-O
vivo, cierre temporal integrado, DIMM exacto, DDR4/PCIe físicos,
inferencia completa y coordinación de múltiples tarjetas.
