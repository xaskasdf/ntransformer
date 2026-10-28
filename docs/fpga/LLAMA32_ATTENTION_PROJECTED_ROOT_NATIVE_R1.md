# Atención completa con proyección O: propietario R1

Estado 20261013T230847Z: **protocolo nativo e interfaces reales aprobados**.
El banco aritmético de esta nueva unión todavía no está implementado ni encolado.
No modifica las tandas O y cuantización → O que ya administra la cola.

## Diseño implementado

`fpga/rtl/ntrpu_attention_projected_root_r1.sv` conecta los bloques reales
`ntrpu_attention_root_r1` y `ntrpu_attention_output_path_r1`.
Retiene las nueve direcciones/extensiones de memoria y la identidad completa
del comando. El resultado síncrono de 2.048 FP32 alimenta al cuantizador.

Un único dueño selecciona el bus AXI. La raíz conserva el bus hasta completar
atención y drenar KV; después lo recibe el camino cuantización/O. Bind y append
permanecen bloqueados hasta consumir la terminación pública, incluso después
de que el cuantizador libere la memoria de la raíz. No se admiten nuevas raíces
durante ese intervalo. El error de raíz se devuelve sin iniciar O.

Una oferta aceptable permanece estable bajo backpressure y cancelación.
Las respuestas con identidad ajena no se consumen. La revocación por epoch o
cancelación es persistente y oculta dirección/tamaño del resultado. Un reset
requiere quiesce coordinado de hijos y memoria externa.

## Pruebas realizadas

Banco nativo con dos sustitutos de protocolo de un beat; **sin atención,
cuantización ni GEMV aritméticos**. Tres semillas, 31 escenarios cada una:
93 trabajos, 87 terminaciones públicas, 90 ACK de raíz y 72 ACK del camino O.
Cubre prioridad de mantenimiento, traspaso síncrono, pausas AXI, errores,
cancelación, respuesta ajena, epoch tardío, recuperación y dos fases de reset.

Diez variantes de RTL defectuosas compilaron y fallaron en simulación:
gating de bind VALID/READY, dueño AXI, descriptor de salida, ACK anticipado,
identidades truncadas en ambos hijos, epoch olvidado, estado HLS perdido y
dirección de lectura alterada. El revisor independiente rechazó 78 controles
de corrupción de trazas, 26 por semilla.

La gráfica de módulos reales también compiló: 34 RTL y tres declaraciones
exactas de interfaces HLS, 37 fuentes. Esa comprobación no ejecuta aritmética.
La atención usa su adaptador de prefetch real; solamente los núcleos generados
se sustituyen por declaraciones vacías durante este filtro sintáctico.

Raíz y O traen dos definiciones del lector AXI. Se selecciona la variante de
entradas `input wire logic` ya calificada por la raíz. Una comparación del
texto normalizado prueba que no cambian cuerpo, nombres ni anchuras; la
integración aritmética todavía debe verificar su uso conjunto.

## Evidencia y siguiente banco

- RTL SHA-256: `c840d071572b6a12048433f8e5b72b5915f3d64f65957c97c253bb7db3d879be`.
- Banco nativo: `20261013T225820Z-V7-attention-projected-root-native-r1`; archivo `5328fa927205fdb3f08efa7fb5e9c50943746ed9d8e7d36c5ed4f3c4814f145e`.
- Revisión independiente: `08e2201009ac2971063974bc0b1bd22c46d225fc04ea5685f151c3daa86642d3`.
- Compilación real: `20261013T230254Z-V7-attention-projected-root-real-compile-r1`; archivo `2c5a0fe2e1d91d1550bcfed459efaf6e5363b07322fbd59114f6e05afcdda0a9`.
- Archivos de cada intento y mutante: verificados en D: y NFS.

La preparación `llama32-attention-projected-root-real-r1-plan.json` fija dos salidas completas con prefijos
de uno y cuatro tokens y una cancelación tras ocho beats Q, seguida de
recuperación. Los 8.192 FP32 de las cuatro referencias de raíz coinciden
exactamente con las entradas del fixture de cuantización/O.

El banco futuro deberá comprobar 64 cabezas, 4.096 resultados O FP64 y
16.384 bytes FP32 finales. KV y ABI1 deberán materializarse mediante AXI;
queda prohibido alimentar al consumidor con la referencia precargada.
El plan conserva las dependencias de ambos padres y exige simulación real,
revisión, síntesis y respaldo. DDR física, ruteo integrado y la capa completa
con residual/normalización/FFN siguen pendientes.
