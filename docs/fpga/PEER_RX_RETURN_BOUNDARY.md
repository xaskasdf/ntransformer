# Fronteras PCIe/HBM del transporte con devoluciones registradas

## Alcance de esta tanda

Estudio lateral sobre el [transporte validado a 200 MHz](PEER_RX_RETURN_CUT.md).
Conserva exactamente su RTL y las dos IP AXI Register Slice. Cambia solamente
el banco de simulación para medir sensibilidad a créditos, latencia y pausas.
Los bloques y validaciones de `docs/fpga/board/` permanecen independientes.
Resultado: `RETURN_BOUNDARY_SWEEP_PASS`; no acredita integración PCIe/HBM.

La entrada compartida de 512 bits a 200 MHz admite como máximo 12,8 GB/s de
payload, antes de burbujas. Dos destinos no duplican ese límite. Los ocho
bancos de 4 KiB —dos dedicados por destino y cuatro de reserva— siguen
sumando 32 KiB; el número de bancos no equivale a créditos PCIe ni HBM.

## Matriz de sensibilidad

45 combinaciones: tres capacidades externas, 1/2/4 obligaciones por destino;
tres tamaños, 64 B/512 B/4 KiB; y los cinco perfiles siguientes. Cada caso
publica 64 trabajos por destino, con tags únicos y comprobación de cada beat.

| Perfil | Demora B | Demora de payload tras fetch | Contrapresión adicional |
| --- | ---: | ---: | --- |
| Rápido | 4 ciclos | 0 | Ninguna |
| B lento | 160 ciclos | 0 | Ninguna |
| Lectura remota demorada | 32 ciclos | 64 ciclos | Ninguna |
| Destino 0 bloqueado | 32 ciclos | 0 | W0 cerrado hasta completar los 64 trabajos de destino 1 |
| AW intermitente | 32 ciclos | 0 | AW cerrado 32 de cada 64 ciclos; W independiente |

Son parámetros sintéticos a 5 ns, no latencias medidas de PCIe/HBM. Una B
puede aceptarse como pronto un ciclo después del vencimiento programado;
su vencimiento nace al completar **ambas** mitades AW/W de ese trabajo.
Cada vencimiento es independiente, evitando serializar artificialmente B.
La demora de payload solo afecta el inicio; después se ofrece un beat por
ciclo y se conserva VALID/data ante bloqueo.

El destino contabiliza una obligación desde su primera AW o W, incluso W
parcial, hasta B. AW y W tienen cursores independientes; una mitad pendiente
no consume otra obligación para el mismo trabajo. La revisión reconstruye
esta unión por tag y verifica capacidad, orden, bytes, tiempos mínimos,
progreso del destino libre y drenaje final. El circuito conserva sus propias
colas y límites: este barrido no los amplía.

## Resultados y decisión

`20261006T084058657426Z-boundary` termina en **113,342 s**. La revisión
independiente acepta 45 casos, 5.760 trabajos, 300.473 ciclos muestreados y
8.970.240 bytes, sin pérdidas ni desbordamiento de créditos. Detecta ocho
controles negativos. Se observan 5.918 beats W anteriores a su AW y se
saturan las tres capacidades en el perfil B lento de 64 B. Los nueve casos
con W0 bloqueada completan los 64 trabajos de destino 1 antes de liberarla.

| Perfil y bloque | 1 crédito/destino | 2 créditos/destino | 4 créditos/destino |
| --- | ---: | ---: | ---: |
| Rápido, 4 KiB | 11,9346 | 11,9346 | 11,9346 |
| B lento, 64 B | 0,1577 | 0,3141 | 0,5958 |
| B lento, 512 B | 1,2077 | 2,3966 | 4,4918 |
| B lento, 4 KiB | 7,2072 | 11,7264 | 11,7264 |
| Payload demorado, 4 KiB | 6,1659 | 6,1659 | 6,1659 |
| AW intermitente, 512 B | 3,1961 | 6,3813 | 8,1665 |

GB/s decimales calculados desde primera publicación hasta última completion,
incluido drenaje; **no son medidas de hardware**. La referencia rápida con
cuatro créditos reproduce exactamente los lotes previos: 659, 1.562 y 8.786
ciclos para 64 B, 512 B y 4 KiB.

Mantener cuatro créditos como opción para trabajos pequeños y dos como
alternativa para el perfil de 4 KiB. No ampliar BRAM en esta tanda. Con B
lento, 4 KiB y dos créditos, solo se ocupan dos bancos; el máximo de ocho
aparece en otros perfiles bloqueados. Con payload demorado, solo dos bancos
están ocupados y todos los créditos comparados empatan: conviene estudiar
prefetch de lecturas remotas antes de añadir almacenamiento. El puerto
actual conserva un único payload activo; este ensayo no implementa varios
fetch remotos concurrentes ni mide los créditos de lectura PCIe.

El primer revisor rechazó únicamente el cambio de ruta `IP_FILE` al crear
el proyecto aislado. Se conserva ese diagnóstico y su respaldo. La revisión
`review-r2/REVIEW.json` exige igualdad de **todas las demás propiedades**,
comprueba el nombre/ruta relativa de la IP y registra ambas rutas. No se
modificó ni repitió la simulación para resolver ese metadato.

## Contrato propuesto para HBM

PG276 expone AXI3 de 256 bits, AWSIZE=5, AWLEN de cuatro bits y direcciones
de 33/34 bits según la capacidad. Por tanto, el adaptador debe conservar los
64 bits de dirección hasta validar/traducir la ventana; truncarlos antes
podría producir alias. Los puertos y restricciones están en
[AMD PG276: Port Descriptions](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions).

Propuesta inicial: INCR alineado, ID cero y orden por destino. Cada beat
de 64 B genera dos de 32 B conservando datos y strobes. Cada fragmento HBM
tiene hasta 16 beats: un bloque de **4 KiB genera ocho ráfagas de 512 B**.
Este cálculo define el contrato inicial; el barrido anterior no incluyó
un adaptador HBM.

La [tanda posterior del adaptador](PEER_RX_HBM_ADAPTER.md) prueba esa
conversión con IP AMD y destino conductual, incluyendo agregación de
respuestas, strobes y rechazos. Su composición con este transporte y el
controlador HBM físico siguen pendientes.

Mantener dos contadores separados: trabajos padres y fragmentos hijos.
Cuatro padres de 4 KiB podrían exigir 32 hijos si se emiten completos sin
esperar respuestas. PG276 permite aceptar 32 escrituras por puerto; ese
límite no reserva capacidad para otros maestros ni obliga a emitirlas todas.
La cola de hijos debe admitir contrapresión y guardar su asociación al padre.
([AMD PG276: AXI Port Details](https://docs.amd.com/r/en-US/pg276-axi-hbm/AXI-Port-Details)).

El padre recibe una única B después de retirar todas las B hijas y consumir
todos sus datos. Su descriptor y crédito de respuesta siguen ocupados hasta
esa finalización. El banco de payload tiene otra vida: puede reutilizarse
cuando todos sus bytes han sido aceptados por almacenamiento independiente
y ningún fragmento necesita releerlos. El núcleo actual devuelve bancos al
terminar W; la integración debe preservar esa transferencia de propiedad.
Un error no permite abandonar solicitudes ya ofrecidas. Propuesta de agregación: DECERR domina
SLVERR, que domina OKAY; respuestas inesperadas enclavan fallo. Una vez
comprometido un padre válido, terminar sus fragmentos, incluso ante `stop`.

## Relojes, publicación y recuperación

Conservar el dominio interno a 200 MHz y definir el CDC hacia el reloj real
del endpoint. La frecuencia/ancho de `axi_aclk` dependen de la configuración
del puente; se extraerán de la IP generada.
([AMD PG194: Clocking](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Clocking)).
HBM distingue relojes de referencia, APB y AXI; los AXI pueden ser asíncronos.
([AMD PG276: Clocking](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking)).
Las demoras de este banco no sustituyen CDC, restricciones generadas ni STA.

Para compartir pesos entre NTRPUs, separar finalización local y publicación
remota. PG194 garantiza orden de transmisión del MemWr al emitir B; de ello
no se deduce que el consumidor remoto ya haya publicado el bloque. La
propuesta exige un ACK remoto con identidad del bloque y generación tras
su recepción y validación, antes de anunciarlo utilizable. BAR proporciona
la ventana de acceso; el catálogo y el protocolo definen la memoria lógica
compartida.
([AMD PG194: Transaction Ordering](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Transaction-Ordering-for-PCIe)).

Preservar la separación entre cerrar admisión, drenar y autorizar reset.
PG276 advierte que activar el reset AXI después de iniciar tráfico desincroniza
interfaz y controlador. No conectar automáticamente el reset del endpoint
a ese reset HBM.
([AMD PG276: Port Descriptions](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions)).

## Criterios antes de conectar controladores reales

1. Fijar FPGA/paquete, IP y modo PCIe, mapa HBM, ventanas y relojes reales.
2. Probar conversión 512→256, fragmentación, extremos de ventana, strobes,
   ocho B por bloque, errores intermedios y respuestas retenidas. Verificar
   ausencia de B padre prematura y contabilizar todos los hijos pendientes.
3. Componer CDC y adaptar la admisión a créditos reales. Incluir pérdida de
   reloj/enlace, `stop`, drenaje y la coordinación de reset del diseño de placa.
4. Repetir síntesis, CDC, STA y comparación de netlist de la composición;
   después medir hardware y P2P. El cierre del núcleo no se transfiere
   automáticamente a esa composición.

## Reproducción

Con Vivado 2026.1 cargado y una salida nueva:

```bash
python3 -B tools/run_ntrpu_return_boundary.py /tanda/output \
  --reference /registered_return_rtl/output --timeout 900
python3 -B tools/check_ntrpu_return_boundary.py /tanda/output \
  --review /tanda/REVIEW.json
```

El ejecutor verifica las fuentes congeladas, usa dos CPU con prioridad baja
y genera manifiestos SHA-256. Evidencia de esta tanda:
`C:\Users\xasko\artifacts\ntrpu-return-boundary\20261006T084058317345Z`.
Respaldo incremental en `D:\AI\artifacts\ntrpu-content-store`.
