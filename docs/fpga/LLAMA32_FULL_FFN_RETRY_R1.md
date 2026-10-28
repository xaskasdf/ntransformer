# FFN Llama real: recuperación del banco XSIM

Fecha: 2026-10-12 UTC. Se conservan los dos intentos fallidos y sus fuentes.
Ninguno acredita una FFN completa con pesos reales.

## Diagnósticos preservados

1. `20261012T005311Z-V7-llama32-full-ffn-r1`: Vivado rechazó la propiedad
   cuyo valor comenzaba con `-testplusarg`. La variante Tcl R2 utiliza
   `set_property -dict [list ...]`, como el banco GEMV ya validado.
   No llegó a compilar/simular el circuito.
   Archivo: `7af972863278cdb351566cfa184fce92c46c111bc96b6edd4adea618008a4214`.
2. `20261012T005755Z-V7-llama32-full-ffn-r2`: compiló y ejecutó, pero
   `$readmemh` sobre `llama_weights[b-1]` produjo tres avisos de índice
   fuera de rango. La primera escritura gate falló en el ciclo 25.255:
   byte recibido `00`, esperado `af`. Se conserva el FAIL y no se atribuye
   al circuito una validación con memoria de prueba incorrectamente cargada.
   Archivo: `cc694125ca0d29956c4304db481d0280c38fa433a85239ea676e8f7c15f660bd`.

Ambos archivos y todos sus miembros tienen certificado local verificado.
Los revisores numéricos y las fuentes ejecutadas permanecen intactos.

## Corrección verificable

El testbench R2 usa tres arreglos planos de 17.829.824 bytes: imagen de
17.825.792 bytes más guarda de 4.032 bytes por matriz. Después de cada
carga recorre la imagen completa y verifica FNV-1a de 64 bits contra los
bytes del fixture cuyo SHA-256 ya está fijado. Este checksum comprueba la
carga del simulador; no sustituye el SHA-256 de procedencia.

El controlador R4 exige los tres registros de carga correctos, rechaza
avisos de límites y mantiene los scoreboards de valores y tráfico AXI.
El DUT, los dos núcleos HLS, los pesos y la referencia numérica no cambian.
La aceptación sigue exigiendo 2.048 resultados FP32, todos los intermedios,
56.754.176 bytes leídos y 155.648 bytes escritos correctos.

La cola dependiente debe usar este nuevo coordinador y conservar su orden:
FFN → relojes comparables → replays optimizados → implementación/readback →
revisión física → replays ruteados → máscaras integradas/síntesis/replay.
Un FAIL detiene las etapas dependientes; nunca se reescribe como PASS.
