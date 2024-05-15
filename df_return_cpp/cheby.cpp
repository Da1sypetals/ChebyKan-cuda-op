#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void cheby_launcher(const float *x, float *cheby, int batch_size, int in_feats, int degree);
void cheby_bwd_launcher(const float *gout, const float *x, const float *cheby, float *grad_x, int batch_size, int in_feats, int degree);

torch::Tensor cheby_cuda_fwd(torch::Tensor x, int degree)
{

    CHECK_INPUT(x);

    const float *x_ptr = x.data_ptr<float>();
    int batch_size = x.size(0);
    int in_feats = x.size(1);

    // create cheby tensor
    torch::Tensor cheby = torch::ones({degree + 1, batch_size, in_feats},
                                      at::device(at::kCUDA).dtype(at::kFloat))
                              .requires_grad_(true);

    float *cheby_ptr = cheby.data_ptr<float>();

    cheby_launcher(x_ptr, cheby_ptr, batch_size, in_feats, degree);

    return cheby;
}

torch::Tensor cheby_cuda_bwd(torch::Tensor gout, torch::Tensor x, torch::Tensor cheby)
{

    CHECK_INPUT(x);
    CHECK_INPUT(cheby);

    const float *gout_ptr = gout.data_ptr<float>();
    const float *x_ptr = x.data_ptr<float>();
    const float *cheby_ptr = cheby.data_ptr<float>();

    int batch_size = x.size(0);
    int in_feats = x.size(1);
    int degree = cheby.size(0) - 1;

    // create grad_x tensor
    torch::Tensor grad_x = torch::zeros_like(x);
    float *grad_x_ptr = grad_x.data_ptr<float>();

    cheby_bwd_launcher(gout_ptr, x_ptr, cheby_ptr, grad_x_ptr, batch_size, in_feats, degree);

    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &cheby_cuda_fwd, "cheby forward");
    m.def("backward", &cheby_cuda_bwd, "cheby backward");
}