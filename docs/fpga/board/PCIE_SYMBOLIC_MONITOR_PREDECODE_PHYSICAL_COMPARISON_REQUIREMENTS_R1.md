# Próxima comparación física E/D: referencia y criterios

## Decisión basada en los archivos actuales

La tanda física 955 usa el monitor predecode anterior a las optimizaciones
de selector RC, bitmap RQ y consumidores posteriores. Su ruta E→D de
17,145 ns de interconexión identifica un problema real de aquella variante,
pero no es una referencia controlada para atribuir diferencias al nuevo
cambio de habilitaciones E/D.

El experimento E/D actual parte de **RQ 984** y conserva C y B de esa
versión. La referencia física correspondiente debe partir del DCP 984:
`924f3c41cedf89364e30201552777ac748b6fa66d087d70f679be7f883a59631`.
Su replay propio 985 ya pasó; no se sustituye por un DCP con los cambios
aislados de suma E, límites B o conteo C.

## Condiciones antes de ejecutar

La cadena vigente conserva prioridad: cierre del replay C, diagnósticos
E/B/C y banco aislado, integración, síntesis, replay y diagnóstico E/D.
Los resultados E/D determinarán la decisión de implementación. Este
documento registra requisitos; todavía no crea un ejecutor físico nuevo.

Para cada miembro de la comparación se necesita:

1. Netlist sintetizada propia revisada y respaldada.
2. Dos copias de su DCP para revisar la inserción BUFGCE: grafo lógico
   idéntico al contraer el buffer, CE constante, puertos/registros
   conservados y reloj exclusivo. Mantener biblioteca y restricciones.
3. Replay funcional de la netlist optimizada exacta, con las 174.436
   filas y sus controles; conservar todos los resultados terminales.
4. `place_design` y `route_design -directive Quick`, mismo Vivado,
   dispositivo, período de 4 ns, E/S cero, dos hilos y límite de 1.800 s.
   Los intentos Default agotados anteriormente no se repiten.
5. Reapertura del DCP, auditorías estructural/temporal/física y replay
   funcional propio del resultado ruteado, con respaldo por tanda.

## Interpretación permitida

Comparar recursos, setup/hold global y por etapas, endpoints, lógica,
interconexión y fanout. Conservar la cobertura de restricciones y los
límites OOC. Reportar también regresiones: una mejora E→D no aprueba por
sí sola el resto del diseño. `Quick` es un diagnóstico comparable, no una
prueba del mejor resultado de implementación posible.

El ruteo 955 permanece como antecedente, separado de la comparación
controlada RQ/E/D. Ni esta comparación ni el replay sin SDF cierran PCIe
nativo, DDR, restricciones de placa, ECAD o inferencia con pesos Meta.
