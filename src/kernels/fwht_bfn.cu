// fwht_bfn.cu
__global__ void fwht_bfn(float* a, int h, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = h * 2;
    int j = tid * stride;

    if (j + h < n) {
        for (int k = 0; k < h; ++k) {
            int idx1 = j + k;
            int idx2 = j + k + h;

            float x = a[idx1];
            float y = a[idx2];

            a[idx1] = x + y;
            a[idx2] = x - y;
        }
    }
}
