#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))

__global__ void cheby_fwd_kernel(const float *x, float *cheby, int batch_size, int in_feats, int degree, int numThreads){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {
        int irow = idx / in_feats;
        int icol = idx % in_feats;
        
        float x_val = x[idx];
        cheby[INDEX3D(1, irow, icol, batch_size, in_feats)] = x_val;
        float cheby_val_z = x_val; // index = i - 1
        float cheby_val_zz = 1;    // index = i - 2

        for(int d = 2; d < degree + 1; d++){
            float new_cheby_val = 2 * x_val * cheby_val_z - cheby_val_zz;
            cheby[INDEX3D(d, irow, icol, batch_size, in_feats)] = new_cheby_val;
            cheby_val_zz = cheby_val_z;
            cheby_val_z = new_cheby_val;
        }
    }
}


__global__ void cheby_bwd_kernel(const float* gout, const float *x, const float *cheby, float* grad_x,
                                 int batch_size, int in_feats, int degree, int numThreads){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {

        int irow = idx / in_feats;
        int icol = idx % in_feats;

        float b0 = 0, b1 = 1;
        float b_z = b1, b_zz = b0; // b(i-1) and b(i-2)

        float x_val = x[idx];

        // grad wrt d=0 is equal to zero
        // here is grad wrt d=1
        float grad_x_val = gout[INDEX3D(1, irow, icol, batch_size, in_feats)];

        for(int d = 2; d < degree + 1; d++){

            // 2a(i-1)
            float b = 2 * cheby[INDEX3D(d - 1, irow, icol, batch_size, in_feats)] 
                      + 2 * x_val * b_z - b_zz;

            grad_x_val += gout[INDEX3D(d, irow, icol, batch_size, in_feats)] * b;

            // finally
            b_zz = b_z;
            b_z = b;
        }

        grad_x[idx] = grad_x_val;


    }
}



void cheby_launcher(const float *x, float *cheby, int batch_size, int in_feats, int degree){
    int numThreads = batch_size * in_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    cheby_fwd_kernel<<<blockSize, threadSize>>>(x, cheby, batch_size, in_feats, degree, numThreads);
}

void cheby_bwd_launcher(const float *gout, const float *x, const float *cheby, float *grad_x,
                        int batch_size, int in_feats, int degree){
    int numThreads = batch_size * in_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    cheby_bwd_kernel<<<blockSize, threadSize>>>(gout, x, cheby, grad_x, batch_size, in_feats, degree, numThreads);
}


