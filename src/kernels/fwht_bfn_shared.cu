// fwht_bfn_shared.cu
#include <cmath>

__global__ void fwht_shared(float* a, int n) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;

    if (tid < n) {
        s_data[tid] = a[tid];
    }
    __syncthreads();

    for (int h = 1; h < n; h *= 2) {
        int j = tid & ~(2 * h - 1);
        int k = tid % h;

        if (j + k + h < n) {
            float x = s_data[j + k];
            float y = s_data[j + k + h];
            s_data[j + k] = x + y;
            s_data[j + k + h] = x - y;
        }
        __syncthreads();
    }

    if (tid < n) {
        a[tid] = s_data[tid] / sqrtf((float)n);
    }
}
