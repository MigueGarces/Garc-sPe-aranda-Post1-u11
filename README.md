# Post-Contenido 1 — CUDA Benchmark CPU vs GPU
## Arquitectura de Computadores — Unidad 11
### Universidad Francisco de Paula Santander — Ingeniería de Sistemas — 2026

---

## Descripción del Entorno

| Elemento        | Detalle                          |
|-----------------|----------------------------------|
| GPU             | NVIDIA Tesla T4 (Google Colab)   |
| CUDA Version    | 12.x                             |
| Sistema Operativo | Ubuntu 20.04 LTS               |
| Compilador      | nvcc (NVIDIA CUDA Compiler)      |
| Comando compilación vectorAdd | `nvcc -O2 -o vectorAdd src/vectorAdd.cu` |
| Comando compilación matMul    | `nvcc -O2 -o matMul src/matMul.cu`       |

---


## Resultados — Multiplicación de Matrices (matMul)

| N (dimensión) | Naïve GPU (ms) | Tiled GPU (ms) | Speedup Tiling |
|---------------|----------------|----------------|----------------|
| 512           | ~30.30         | ~0.46          | ~66.33x        |
| 1024          | ~9.19          | ~5.81          | ~1.58x         |

---

## Análisis de Resultados

### ¿Por qué la GPU es más rápida que la CPU para N grande?

Cuando el vector tiene millones de elementos, la GPU puede sumarlos todos casi
al mismo tiempo porque tiene miles de núcleos trabajando en paralelo. La CPU,
aunque tiene una frecuencia de reloj muy alta, procesa los elementos uno por uno
. Por eso, para N = 16 millones de elementos,
la GPU termina en menos de 2 ms mientras que la CPU necesita unos 35 ms,
logrando un speedup de casi 20 veces.

### ¿Por qué el tiempo total con memcpy puede ser mayor que la CPU para N pequeño?

Aunque el kernel GPU es muy rápido, mover los datos desde la RAM de la CPU
a la memoria de la GPU (cudaMemcpy HostToDevice) tiene un costo adicional.
Para vectores pequeños ese costo de transferencia puede ser mayor que el tiempo
que ahorra el kernel, haciendo que la GPU en total sea más lenta que la CPU.
Esto demuestra que la GPU no siempre gana: solo vale la pena usarla cuando
el volumen de datos es lo suficientemente grande para amortizar el costo de la
transferencia.

---
