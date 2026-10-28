# Brandon: FFN con pesos reales y referencia GPU

La prueba ejecuta las 12 FFN únicas de [Brandon-Tiny-10M-Instruct](https://huggingface.co/xaskasdf/brandon-tiny-10m-instruct/tree/ca6329de991a1cfc17b2a7a1aaae2cc38c08577a) sobre dos unidades NTRPU simuladas. Compara con NumPy o PyTorch FP32 y con una implementación Q8 independiente. Es una FFN aislada con pesos reales y entradas sintéticas; todavía no genera texto ni mide calidad del modelo.

## Modelo y procedencia

Revisión fijada: `ca6329de991a1cfc17b2a7a1aaae2cc38c08577a`. El script verifica SHA-256 de `config.json` y `model.safetensors` antes de leerlos. No importa código remoto ni carga pickle.

El archivo contiene **30.019.416 elementos BF16 almacenados**, pero **10.706.776 parámetros únicos**. Se comprobó igualdad de las copias `_unique_layers.i`, `layers.2i` y `layers.2i+1`, además de embeddings y salida compartidos. El tamaño que muestra Hugging Face incluye esas copias; no corresponde a otra arquitectura de 30M.

La [implementación original, revisión cd56eb1](https://github.com/xaskasdf/brandon-tiny/blob/cd56eb1093b4b1da986cefc11c0ad79c9d358099/src/model.py), define `w2(silu(w1(x)) * w3(x))`. Usamos gate=`w1`, up=`w3`, down=`w2`, sin dropout en inferencia. El modelo completo también requiere DenseFormer, value residual, register tokens y block sharing. Convertirlo como un Llama convencional perdería esas operaciones; la integración GGUF sigue pendiente.

## Flujo de cálculo

Cada FFN tiene forma `256 → 720 → 256`. Gate/up se reparten en 360 filas por unidad. `submit_swiglu()` consume sus salidas locales, y DMA reúne la activación en la unidad 1. Down usa 736 columnas: las últimas 16 columnas y activaciones son cero, para completar bloques Q8 de 32. La salida vuelve a la unidad 0. No hay lecturas/escrituras host de activaciones intermedias.

Por muestra se ejecutan cinco GEMV, dos SwiGLU y cuatro DMA: **557.056 MAC**, **591.872 bytes de pesos locales**, **3.488 bytes entre unidades** y **4.928 bytes DMA totales**. El padding añade 4.096 MAC frente a la FFN original. Solo se materializan 638.976 bytes de memoria simulada. La carga inicial de pesos queda fuera del contador DMA; estos valores no son medidas de ancho de banda ni latencia.

SwiGLU usa sigmoid estable y producto en double, con redondeo final FP32. Es un modelo CPU, aún sin top HLS propio. Su comando comparte épocas, pins, cancelación y completions; exige operandos locales y admite hasta 1 MiB de salida. Un error numérico o de recursos deja intacto el destino. Su completion informa bytes de cómputo, cero DMA y cero MAC; los nuevos contadores son `swiglu_commands` y `swiglu_elements`.

Q8 conserva el formato de 34 bytes por bloque: escala FP16 little endian y 32 INT8. El exportador experimental usa `maxabs/127`, redondeo de empates alejándose de cero y escala almacenada FP16. Las activaciones se cuantizan por bloques; las escalas/acumulación permanecen double en CPU. El formato binario de prueba `NTRFFN01` contiene dimensiones, tres matrices Q8, entradas FP32 y dos referencias FP32; no es GGUF ni un formato público de modelos.

## Repetir localmente

Compilar primero según [SIMULATOR.md](SIMULATOR.md). Para la referencia NumPy basta Python con NumPy; PyTorch solo se usa con `--device cpu` o `--device cuda`.

```bash
python -m pip install numpy==2.5.3 huggingface_hub==2.0.0
hf download xaskasdf/brandon-tiny-10m-instruct config.json model.safetensors \
  --revision ca6329de991a1cfc17b2a7a1aaae2cc38c08577a \
  --local-dir build/models/brandon-tiny-ca6329d
python tools/run_ntrpu_brandon.py \
  --model-dir build/models/brandon-tiny-ca6329d \
  --exe build/ntrpu-sim/src/ntrpu/ntrpu_ffn_fixture \
  --output-dir build/ntrpu-brandon/run-01 --device numpy
python tests/test_ntrpu_brandon.py
```

En Windows/Visual Studio, el ejecutable es `build/ntrpu-sim/src/ntrpu/Release/ntrpu_ffn_fixture.exe`; usar los argumentos en una sola línea en PowerShell. Preferir un entorno virtual. La validación local GPU utilizó `build/ntrpu-reference-venv/Scripts/python.exe`, Python 3.12.13, NumPy 2.5.3 y PyTorch 2.13.0+cu126, instalado desde el [índice oficial de PyTorch](https://pytorch.org/get-started/previous-versions/).

`--layer 0` selecciona una FFN única; por defecto se prueban las 12, con ocho vectores por FFN. Las entradas son normales aleatorias, normalizadas a RMS uno, con semilla `20261001 + capa`; no proceden de prompts ni del transformer. El directorio de salida debe ser nuevo. Se guardan fixtures y `report.json` con revisiones, hashes, versiones y errores. Pedir CUDA sin GPU disponible produce error; no cambia silenciosamente a CPU.

## Resultados del 2026-10-01

Pasaron **45/45 pruebas CTest en Windows/MSVC 19.51 y Linux/WSL/GCC 13.3, Release**, más cuatro pruebas Python del exportador. Incluyen solapamiento, salida atómica ante overflow, localidad, permisos, pins, cancelación, pérdida de unidad, presupuesto y archivos truncados o inválidos. Ninguna prueba básica descarga el modelo ni necesita GPU.

Con 96 evaluaciones de FFN por backend, el L2 relativo frente a FP32 estuvo entre **0,8716 % y 1,5550 %**, tanto con NumPy como con **RTX 3090 de 24 GB / PyTorch CUDA, TF32 desactivado**. La salida C++ coincidió exactamente con los valores FP32 de la referencia Q8 NumPy en esas muestras. Se conserva el [informe GPU](validation/brandon-cuda-20261001.json). El límite de regresión es 5 % de L2 por FFN; la referencia Q8 permite `atol=1e-6`, `rtol=1e-5`. Estos umbrales no garantizan calidad del modelo ni equivalencia con aritmética RTL futura.

## Referencia GPU opcional en Lambda

El uso principal del cloud en este proyecto es [Vivado/Vitis para síntesis y co-simulación](CLOUD_HLS.md). Las instrucciones siguientes son únicamente para repetir la referencia PyTorch; no preparan el entorno AMD.

La A100 propuesta tiene capacidad suficiente para esta referencia. La [tarifa pública consultada el 2026-10-01](https://lambda.ai/pricing) indica **A100 SXM 40 GB, 30 vCPU, 220 GiB RAM, 512 GiB SSD, US$1,99/h más impuestos**; comprobar configuración/disponibilidad al contratar. Ocho horas equivalen a US$15,92 antes de impuestos y almacenamiento adicional. Para este modelo pequeño, la RTX 3090 local ya permitió completar la prueba.

Lambda proporciona una instancia donde ejecutar Docker; sus [instrucciones de entorno](https://docs.lambda.ai/public-cloud/on-demand/managing-system-environment/) documentan Docker y NVIDIA Container Toolkit. Desde una copia de **este árbol de trabajo** en Linux, incluidos los archivos NTRPU nuevos:

```bash
mkdir -p build/ntrpu-cloud
docker build -t ntrpu-reference docker/ntrpu-reference
docker run --rm --gpus all \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/build/ntrpu-cloud:/work" \
  ntrpu-reference
```

El Dockerfile fija la imagen PyTorch por digest. El script compila NTRPU **sin CUDA**, ejecuta sus pruebas, descarga la revisión fija y utiliza CUDA únicamente para la referencia. Los resultados quedan en `build/ntrpu-cloud/results/`; el montaje sobrevive al contenedor, pero no garantiza persistencia al eliminar la VM. Guardar el repositorio y los informes en almacenamiento persistente o copiarlos fuera antes de terminar la instancia. El contenedor no incluye Vitis ni modifica drivers/BAR/NVMe. **La receta Docker está preparada; no se construyó aquí ni se contrató una instancia Lambda.**

La configuración de la estación de síntesis AMD, incluidos recursos, instalación y primer trabajo remoto, se desarrolla en [CLOUD_HLS.md](CLOUD_HLS.md).

El siguiente hito de software es ejecutar un bloque completo con normalización, atención y sus particularidades, y luego comparar logits sobre entradas reales. En hardware siguen pendientes síntesis/co-simulación, escalas y activación implementadas en HLS, y las interfaces DDR/PCIe.
