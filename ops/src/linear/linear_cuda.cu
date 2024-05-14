#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)

__global__ void linear_kernel(const float *x, const float *w, const float *b, float *result, int batch_size, int in_feats, int out_feats){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        for (int i = 0; i < out_feats; i++) {
            float r = 0;
            for (int j = 0; j < in_feats; j++) {
                r += x[idx * in_feats + j] * w[i * in_feats + j];
            }
            result[idx * out_feats + i] = r + b[i];
        }
    }
}


__global__ void backward_weight_kernel(const float* gout, const float* x, float* result, int batch_size, int in_feats, int out_feats){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_feats) {
        for(int i = 0; i < in_feats; i++){
            float r = 0;
            for(int b = 0; b < batch_size; b++){
                
                r += gout[b * out_feats + idx] * x[b * in_feats + i];

            }
            result[idx * in_feats + i] = r;
        }
    }

}

void linear_launcher(const float *x, const float *w, const float *b, float *result, int batch_size, int in_feats, int out_feats){
    dim3 blockSize(DIVUP(batch_size, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    linear_kernel<<<blockSize, threadSize>>>(x, w, b, result, batch_size, in_feats, out_feats);
}

void backward_weight_launcher(const float *gout, const float *x, float *result, int batch_size, int in_feats, int out_feats){

    dim3 blockSize(DIVUP(out_feats, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    backward_weight_kernel<<<blockSize, threadSize>>>(gout, x, result , batch_size, in_feats, out_feats);

}


