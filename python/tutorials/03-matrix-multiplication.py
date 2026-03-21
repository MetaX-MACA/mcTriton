"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves
performance on par with mcBLAS.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.
"""

import torch
import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_maca():
    return triton.runtime.driver.active.get_current_target().backend == "maca"

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_maca_autotune_config_tn():
    return [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': 'storeCoalesce'},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'pipeline':'cpasync', 'scenario': ''},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'pipeline':'cpasync', 'scenario': ''},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                      num_stages=3, num_warps=4),
    ]

def get_maca_autotune_config_tt():
    return [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                       num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                       num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                       num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'pipeline':'cpasync', 'scenario': ''},
                       num_stages=4, num_warps=4),
    ]


matmul_mod = None
def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    elif is_maca():
        if matmul_mod == "TN":
            return get_maca_autotune_config_tn()
        return get_maca_autotune_config_tt()
    else:
        assert False, "Not valid config"


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.heuristics({"MASK_K_ONLY": lambda args: (args["K"] % args["BLOCK_SIZE_K"] != 0)
                                   and (args["M"] % args["BLOCK_SIZE_M"] == 0) and (args["N"] % args["BLOCK_SIZE_N"] == 0)})
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        MASK_K_ONLY: tl.constexpr,
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if MASK_K_ONLY:
            a_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
rtol = 1e-4
if torch.allclose(triton_output, torch_output, rtol=1e-2, atol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = False
ref_lib = 'cuBLAS' if is_cuda() else 'mcBLAS'
configs = []
for fp8_inputs in [False]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if matmul_mode == "TN":
        b = b.transpose(0, 1)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


matmul_mode = "TN"
benchmark.run(show_plots=True, print_data=True)

matmul_mode = "TT"
benchmark.run(show_plots=True, print_data=True)

'''
C500 TN
matmul-performance-fp16:
         M       N       K      mcBLAS      Triton
0    256.0   256.0   256.0    2.473057    2.221559
1    384.0   384.0   384.0    7.372800    6.912000
2    512.0   512.0   512.0   13.797053   13.107200
3    640.0   640.0   640.0   23.272727   26.597404
4    768.0   768.0   768.0   36.111675   30.508137
5    896.0   896.0   896.0   48.867060   44.957694
6   1024.0  1024.0  1024.0   66.576254   61.230717
7   1152.0  1152.0  1152.0   87.823057   81.807778
8   1280.0  1280.0  1280.0  110.702698  103.696206
9   1408.0  1408.0  1408.0   92.013097   79.011245
10  1536.0  1536.0  1536.0  109.311014   96.297797
11  1664.0  1664.0  1664.0  132.336939  116.114992
12  1792.0  1792.0  1792.0  150.360184  135.414749
13  1920.0  1920.0  1920.0  135.198044  115.200003
14  2048.0  2048.0  2048.0  155.344592  132.104056
15  2176.0  2176.0  2176.0  160.989176  151.876583
16  2304.0  2304.0  2304.0  156.385415  132.710398
17  2432.0  2432.0  2432.0  174.499773  150.237777
18  2560.0  2560.0  2560.0  194.468840  166.758272
19  2688.0  2688.0  2688.0  149.049343  151.278396
20  2816.0  2816.0  2816.0  173.244126  165.676006
21  2944.0  2944.0  2944.0  179.105240  156.593970
22  3072.0  3072.0  3072.0  195.252085  168.646626
23  3200.0  3200.0  3200.0  168.421047  161.820475
24  3328.0  3328.0  3328.0  177.100355  170.091665
25  3456.0  3456.0  3456.0  191.046364  166.659576
26  3584.0  3584.0  3584.0  204.935369  175.359125
27  3712.0  3712.0  3712.0  196.648305  173.282463
28  3840.0  3840.0  3840.0  209.653080  181.224091
29  3968.0  3968.0  3968.0  185.024917  179.381016
30  4096.0  4096.0  4096.0  199.062255  187.980005

C500 TT
matmul-performance-fp16:
         M       N       K      mcBLAS      Triton
0    256.0   256.0   256.0    2.340571    1.724632
1    384.0   384.0   384.0    7.372800    5.084690
2    512.0   512.0   512.0   14.563555   13.443283
3    640.0   640.0   640.0   26.256411   18.123894
4    768.0   768.0   768.0   38.466782   31.043368
5    896.0   896.0   896.0   53.521067   45.320259
6   1024.0  1024.0  1024.0   66.576254   62.137839
7   1152.0  1152.0  1152.0   87.182017   81.807778
8   1280.0  1280.0  1280.0  110.702698  103.044023
9   1408.0  1408.0  1408.0   86.194085   77.882512
10  1536.0  1536.0  1536.0  105.247401   94.058315
11  1664.0  1664.0  1664.0  125.420375  112.839017
12  1792.0  1792.0  1792.0  147.402285  132.618576
13  1920.0  1920.0  1920.0  122.880002  110.371263
14  2048.0  2048.0  2048.0  141.879207  128.561038
15  2176.0  2176.0  2176.0  162.287486  147.696494
16  2304.0  2304.0  2304.0  156.899000  129.298363
17  2432.0  2432.0  2432.0  174.907168  145.378855
18  2560.0  2560.0  2560.0  194.180736  161.817284
19  2688.0  2688.0  2688.0  159.550179  146.742969
20  2816.0  2816.0  2816.0  177.293521  161.984052
21  2944.0  2944.0  2944.0  164.069234  150.562036
22  3072.0  3072.0  3072.0  179.898657  164.006090
23  3200.0  3200.0  3200.0  169.873927  155.765130
24  3328.0  3328.0  3328.0  179.082824  167.714140
25  3456.0  3456.0  3456.0  190.482149  161.323791
26  3584.0  3584.0  3584.0  204.702089  172.665177
27  3712.0  3712.0  3712.0  181.631536  166.218549
28  3840.0  3840.0  3840.0  192.668988  178.015287
29  3968.0  3968.0  3968.0  187.584835  172.411078
30  4096.0  4096.0  4096.0  199.506099  183.545616
'''