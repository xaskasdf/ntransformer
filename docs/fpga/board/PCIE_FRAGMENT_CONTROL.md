# Control del ingreso de fragmentos

Perfil experimental `NTRF`, versión `0x00010000`, en
`ntrpu_pcie_fragment_control.sv`. Corresponde al formato
[NTRPUF01](PCIE_INGRESS_FRAGMENTS.md) y a su
[ensamblador](PCIE_FRAGMENT_INGRESS.md). El perfil DWORD `NTRI` conserva su
módulo y contrato. No son intercambiables.

## Identificación y comandos

AXI-Lite usa direcciones de byte alineadas a cuatro. Los registros
`0x00`, `0x04` y `0x08` devuelven `0x4652544e` (NTRF en little-endian),
`0x00010000` y `0x0000001f`. Los bits de capacidades 0–4 anuncian,
respectivamente: barrera externa obligatoria, formato NTRPUF01, identidad
de 64 bits, captura conjunta de telemetría y contadores de rechazo de
32 bits con saturación en el ensamblador.

El descriptor de ocho palabras (`0x20–0x3c`) contiene época, etiqueta,
ventana y destino, cada uno con palabra baja primero. Conserva las reglas
de [control DWORD](PCIE_INGRESS_PLAN.md): descriptor inmutable mientras
ARM espera o la reserva está activa; época estrictamente creciente;
retiro mediante las cuatro palabras de identidad en `0x40–0x4c`.

Escribir `0x10` solicita una operación: 0 no hace nada, 1 encola ARM,
2 solicita STOP, 4 reconoce el resultado y **8 captura la telemetría**.
No son bits combinables. Se aplican WSTRB y las comprobaciones existentes
de estado e identidad. CAPTURE está permitido también durante una reserva
o un fallo de protocolo; no modifica ni reconoce el resultado, no borra
contadores y no elimina el fallo.

## Lectura coherente de rechazos

| Dirección | Lectura |
| --- | --- |
| `0x90` | `rejected_frames` capturado |
| `0x94` | `rejected_identity` capturado |
| `0x98` | Código en bits 3:0; fallos del decodificador en bits 10:4 |
| `0x9c`, `0xa0` | Magic del formato: `0x5052544e`, `0x31304655` |
| `0xa4` | 64 bytes por trama completa y línea autorizada |
| `0xa8` | 32 bytes útiles por trama |
| `0xac` | Bit 0: existe una captura desde el reset |

Estos registros son de solo lectura. El host escribe CAPTURE, espera su
respuesta B y después lee el banco; debe serializar las capturas entre
lectores. Las tres magnitudes se toman en el mismo flanco de reloj, con
los valores anteriores a ese flanco si el ensamblador actualiza allí sus
contadores. Permanecen congeladas hasta otra captura. Cada respuesta R
conserva su valor mientras READY está bajo, incluso si se reemplaza la
captura entretanto. Antes de la primera captura, el banco y su bit válido
valen cero. ARM y ACK no borran la muestra.

## Composición y alcance

`ntrpu_pcie_fragment_managed` conecta este control al ensamblador mediante
`ntrpu_pcie_fragment_guarded` y `ntrpu_pcie_fragment_peer`. El receptor y
el ensamblador cambian de época conjuntamente. El guardián de capacidad
verifica los 64 bits del destino antes de reducirlo al ancho de memoria.

Las barreras de aislamiento siguen siendo entradas externas de confianza;
el control no las fabrica. Reset frío y coordinado. La evidencia de esta
variante debe calificarse por separado: la aprobación del perfil DWORD no
demuestra la composición de fragmentos, CDC, MIG ni cierre físico.

## Validación

`20261005T001053Z-V7-ffn-supervised-fragment-control` aprueba 12 escenarios
con semillas 1, 7 y 42 en RTL y netlist sintetizada. Cada ejecución realiza
128 escrituras, 52 lecturas y tres ARM/retiros. El revisor reconstruye los
descriptores, las identidades de retiro y todas las peticiones y respuestas
contra una referencia independiente. Coinciden eventos y ciclos entre RTL
y netlist. Se detectan nueve mutaciones RTL y 58 alteraciones de evidencia.

Incluye lectura retenida durante otra captura, valores altos de contadores,
lectura congelada pese a nuevos rechazos y captura después de un fallo de
protocolo. El valor `0xffffffff` lo aporta un BFM: **no prueba la saturación
del contador del ensamblador**. Con semilla 1 termina en 9,240 µs simulados;
hay 384 ciclos de espera B, 244 R y 217 ARM. Herramienta: 256,800 s.

Síntesis VU5P: 839 LUT, 892 FF, sin BRAM/URAM/DSP, latches ni cajas negras.
Reloj de 5 ns; WNS interno estimado 2,807 ns y WHS 0,066 ns. Hay 518 entradas
y 302 salidas sin presupuesto de E/S, sin estimación de skew `HD.CLK_SRC`:
no representa cierre temporal físico. Se conservan tres avisos de síntesis/
OOC y once de entorno `LIBRARY_PATH`; los positivos no tienen avisos de
compilación o simulación. El DRC OOC no informa comprobaciones fallidas.

Respaldo local/NFS verificado el 2026-10-05 a las 00:16:43 UTC: 381 archivos,
3.415.233 bytes, SHA-256
`3c16deaa221a6088da382e7e8b093f438524db5cb6a5a9c254c085dc56316f75`.
El control aislado usa BFMs para disponibilidad, resultados y barreras.

## Evidencia de la composición

`20261005T001706Z-V7-ffn-supervised-fragment-managed` aprueba 21 reservas
con semillas 1, 7 y 42 en RTL y netlist. Por ejecución: 46 tramas, 39
aceptaciones, siete rechazos (tres de identidad), 15 escrituras de memoria
de 64 bytes, 21 resultados, 976 escrituras de control y 197 lecturas.
Se rechazan 128 operaciones de control. La referencia reconstruye los
paquetes y los bytes publicados, y coteja todas las operaciones MMIO.
Coinciden eventos y ciclos entre RTL y netlist. Tres mutaciones RTL
y 265 alteraciones de evidencia son detectadas.

El banco usa épocas y etiquetas con bits altos, duplicados idénticos y
conflictivos, una trama antigua al reutilizar la ventana, huecos, STOP,
SLVERR/DECERR del destino y tres direcciones fuera de capacidad. Estas
últimas no generan tráfico de memoria. Retener B durante 100 ciclos
impide retirar o rearmar la reserva. STOP se **inicia por MMIO** en cinco
fases del puente; su aplicación ocurre después de la latencia del control,
por lo que no se afirma una inyección instantánea en cada fase.

Las capturas se leen por el control conectado al ensamblador; no son
contadores aportados por un BFM. Con semilla 1 hay 60 ciclos de espera AW
y 90 W en el destino, 184 B de ingreso y 2.331 de resultado. Termina en
48,425 µs simulados. Herramienta: 230,624 s. Los 960 bytes de destino son
escrituras observadas por un BFM; todavía no hay lectura DDR en esta tanda.

Síntesis VU5P: 2.591 LUT (298 como LUTRAM), 4.067 FF, sin BRAM/URAM/DSP,
latches ni cajas negras. Con reloj de 5 ns, WNS interno estimado 2,012 ns
y WHS 0,066 ns. Permanecen 843 entradas y 1.289 salidas sin presupuesto de
E/S. Ocho avisos revisados: payload del receptor en registros, cinco RAM
inalcanzables del lado de lectura eliminado, falta de `HD.CLK_SRC` y DRC
OOC parcial. Cinco avisos de entorno `LIBRARY_PATH`. No hay avisos de
compilación o simulación en los positivos. No es cierre temporal físico.

Respaldo local/NFS verificado el 2026-10-05 a las 00:24:55 UTC: 230 archivos,
8.654.280 bytes, SHA-256
`0342229f14619d9f162bbfb673bf0ade889153ab44de88da736aa43989d4c05b`.
El [cruce de ambos dominios y lectura de retorno con MIG frío](PCIE_FRAGMENT_HOST.md)
ya aprueba los perfiles de host rápido y lento. Siguen la netlist de esa
composición, su revisión CDC/temporal y la integración XDMA nativa.
La barrera externa y el reset coordinado siguen siendo requisitos.
