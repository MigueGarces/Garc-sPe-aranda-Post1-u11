
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 16

/* ── Kernel naïve: sin shared memory ── */
__global__ void matMulNaive(const float *A, const float *B,
                             float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

/* ── Kernel con tiling y shared memory ── */
__global__ void matMulTiled(const float *A, const float *B,
                             float *C, int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        /* Cargar tile de A */
        if (row < n && t * TILE + threadIdx.x < n)
            sA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        /* Cargar tile de B */
        if (col < n && t * TILE + threadIdx.y < n)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

/* ── Multiplicación CPU referencia ── */
void matMulCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

void runBenchmark(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A   = (float*)malloc(bytes);
    float *h_B   = (float*)malloc(bytes);
    float *h_C   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)(i % 10) * 0.1f;
        h_B[i] = (float)(i % 7)  * 0.2f;
    }

    /* Referencia CPU */
    matMulCPU(h_A, h_B, h_ref, n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    /* Naïve */
    cudaEventRecord(start);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float naive_ms = ms;

    /* Tiled */
    cudaEventRecord(start);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float tiled_ms = ms;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    /* Verificar */
    int errors = 0;
    for (int i = 0; i < n * n; i++)
        if (fabs(h_C[i] - h_ref[i]) > 1e-3f) errors++;

    printf("N=%d | Naive: %.2f ms | Tiled: %.2f ms | Speedup: %.2fx | Errores: %d\n",
           n, naive_ms, tiled_ms, naive_ms / tiled_ms, errors);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
}

int main() {
    printf("=== Benchmark Multiplicacion de Matrices ===\n");
    runBenchmark(512);
    runBenchmark(1024);
    return 0;
}
