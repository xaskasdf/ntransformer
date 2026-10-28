# Contrato R7: propiedad RC predecodificada

Preparado el 2026-10-08 antes de ejecutar. Base: productor R6 y selector R1;
R5 conserva la referencia de mejor WNS. R6 pasó RTL, síntesis auditada y
su propia netlist. Seguir [el diagnóstico y las dependencias](PCIE_INVENTORY_RQ_ADMISSION_FOLLOWUP.md).

## Cambio propuesto

El selector R2 obtiene propiedad por encabezado desde la tabla registrada.
Un SOP anterior del mismo tag, terminal y con propiedad inicial verdadera
inhibe la propiedad actual. Los antecedentes se calculan sin consultar
liberaciones de otros encabezados. Una decisión compartida alimenta el
selector de ascendencia y el productor R7. No cambiar aritmética RC,
admisión RQ, latencia, capacidad, framing ni criterio de fault.

Conservar decisiones procedurales de cuatro estados: propiedad X/Z se
mantiene; un antecedente desconocido no libera ni suprime propiedad. SOP
duplicados representan un solo encabezado físico y un índice desconocido
no genera SOP. La liberación sucede en SOP, incluso con descriptor erróneo;
el contexto retenido y su EOP no liberan otra vez el tag. RC precede a RQ.

## Bancos y aceptación

- Conservar el banco selector de 89.632 entradas. Ampliar el modelo de
  eventos a 155.168 entradas con propiedad, liberaciones y tags; comprobar
  que las entradas previas y sus 800 bits de salida permanecen iguales.
- Banco del selector real contra referencia R1 congelada y recurrencia
  independiente: 1.153.296 comprobaciones, 994.288 con inyección X/Z.
  Incluye las 15 particiones de igualdad entre cuatro tags y las 65.536
  combinaciones de propiedad/terminal 0/1/X/Z, más 943 bits de entrada
  perturbados con X y Z en ocho contextos. No constituye prueba formal.
- Integración nueva de 46.356 filas/3.872 escenarios: particiones de tags,
  asignaciones, terminales, reutilización RQ, contexto retenido y discontinue.
  Comparar todas las tablas y estados posteriores a fallos con la referencia
  de inventario R7, distinta del nuevo productor R7. Sin inyectar estado DUT.
- Conservar banco principal 22.339/68, siete mutantes, suplementos RC
  45.056/6.624 X/Z y 544/60, RQ 615.536/5.712 X/Z y 8.850/240.
- Sólo tras revisión completa: síntesis OOC con VU5P/4 ns/RuntimeOptimized,
  auditoría DCP y replay de la netlist propia frente al RTL binario,
  incluyendo 22.330/67 y los tres suplementos de integración.

Registrar área, WNS/TNS, setup/hold y peor ruta frente a R6, R5 y original.
No relajar constraints ni ocultar advertencias. Respaldar todos los intentos
con hashes locales/NFS. No acredita timing físico, enlace nativo, reset de
50 ms, DIMM exacto, cuatro DDR/512 GiB, inferencia real o ECAD.
