# E/D con habilitación por fila: replay propio pendiente

## Dependencia y propósito

Ejecutar solamente después de la síntesis E/D revisada, comparada con
RQ 984 y respaldada. El preparador extrae la netlist funcional exacta del
archivo certificado de esa síntesis y enlaza su SHA-256 al banco. No toma
una netlist anterior ni reutiliza resultados de otra variante.

El monitor E/D conserva C RQ y B original. Este replay comprueba que el
hardware binario sintetizado conserva las observaciones públicas de la
referencia; el tratamiento X/Z corresponde a las pruebas RTL previas.

## Cobertura y aceptación

- 174.436 filas binarias nuevas en 16 bancos; 447 bits públicos comparados.
- Diez perturbaciones quiet y sus restauraciones: veinte muestras.
- Seis alteraciones de entrada deben fallar en el primer testigo previsto.
- Los 34 controles de evidencia del revisor deben rechazar logs alterados.
- Se preservan referencias independientes, contabilidad y drenado original.

El Tcl y el banco derivan del replay RQ 985, con nombres específicos E/D.
Las funciones del revisor permanecen idénticas. Se compilan 19 fuentes
del banco más la netlist propia y los modelos del proveedor. El límite es
3.000 s y dos hilos. No se usa SDF ni se aprueba timing físico.

## Evidencia y siguiente paso

`build/cloud-setup/20261002-board/pcie-symbolic-monitor-predecode-row-enable-binary-r1-preparation.json`
registra seis productos y sus recetas exactas. SHA-256:
`bb2e46ac07f0ccfe187422437e70b7a18a28a1fe92736e657b5fc4fda761c0bd`.
La configuración congelada y el enlace a la síntesis se crean al cumplirse
la dependencia. Las comprobaciones SSH viajan por entrada estándar UTF-8.

Todos los resultados terminales se respaldan en D: y NFS. Si pasan la
revisión, corresponde comparar las rutas E→D y el coste de los registros
añadidos antes de decidir una nueva colocación/ruteo. Todavía no hay
mediciones físicas de esta variante.
