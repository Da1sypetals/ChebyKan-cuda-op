#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void linear_launcher(const float *x, const float *w, const float *b, float *result,
                     int batch_size, int in_feats, int out_feats);

void backward_weight_launcher(const float *gout, const float *x, float *result,
                              int batch_size, int in_feats, int out_feats);

void linear_cuda_fwd(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor result)
{

    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);

    const float *x_ptr = x.data_ptr<float>();
    const float *w_ptr = w.data_ptr<float>();
    const float *b_ptr = b.data_ptr<float>();
    float *res_ptr = result.data_ptr<float>();

    int batch_size = x.size(0);
    int in_feats = x.size(1);
    int out_feats = w.size(0);

    linear_launcher(x_ptr, w_ptr, b_ptr, res_ptr, batch_size, in_feats, out_feats);
}

void linear_cuda_bwd_weight(at::Tensor gout, at::Tensor x, at::Tensor result)
{

    CHECK_INPUT(gout);
    CHECK_INPUT(x);
    CHECK_INPUT(result);

    const float *gout_ptr = gout.data_ptr<float>();
    const float *x_ptr = x.data_ptr<float>();
    float *res_ptr = result.data_ptr<float>();

    int batch_size = gout.size(0);
    int out_feats = gout.size(1);
    int in_feats = x.size(1);

    backward_weight_launcher(gout_ptr, x_ptr, res_ptr, batch_size, in_feats, out_feats);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &linear_cuda_fwd, "linear forward");
    m.def("backward_weight", &linear_cuda_bwd_weight, "linear forward");
}