#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name("MTL0", nullptr);
    if (!be) { printf("no MTL0\n"); return 1; }
    ggml_init_params ip = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 64, 1);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * out = ggml_get_rows(ctx, a, idx);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    uint8_t blk[36];
    blk[0] = 0x38; blk[1] = 0x40; blk[2] = 0x01; blk[3] = 0xFF; // 1.0*0.5, 2*0.5, 2^-10, 240
    for (int i = 0; i < 32; i++) blk[4+i] = (uint8_t)((i & 7) | (((i+1) & 15) << 4));
    ggml_backend_tensor_set(a, blk, 0, 36);
    int32_t r = 0; ggml_backend_tensor_set(idx, &r, 0, 4);
    ggml_backend_graph_compute(be, gf);
    float y[64]; ggml_backend_tensor_get(out, y, 0, sizeof(y));
    std::vector<float> ref(64);
    ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(blk, ref.data(), 64);
    for (int i = 0; i < 64; i++) printf("%2d metal=%g cpu=%g\n", i, y[i], ref[i]);
    return 0;
}
