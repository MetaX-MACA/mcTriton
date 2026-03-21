"""
Fused Softmax
=============

In this tutorial, you will write a fused softmax operation.

In doing so, you will learn about:

* The benefits of kernel fusion for bandwidth-bound operations.

* Reduction operators in Triton.

"""

import torch
import triton
import triton.language as tl
from triton.runtime import driver


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y


torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 60)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)

'''
C500 softmax-performance:
         N       Triton        Torch
0    256.0   145.853529   379.697051
1    384.0   207.627460   488.634685
2    512.0   275.814488   632.012042
3    640.0   320.416217   693.527823
4    768.0   382.721969   756.173017
5    896.0   441.291893   802.003472
6   1024.0   502.102868   864.658977
7   1152.0   540.008876   769.512503
8   1280.0   599.593639   851.329918
9   1408.0   653.725090   906.167276
10  1536.0   702.088654   961.306184
11  1664.0   756.287765   989.441334
12  1792.0   803.151928  1023.074291
13  1920.0   849.749118  1043.716681
14  2048.0   894.308805  1096.442606
15  2176.0   825.561790   938.486752
16  2304.0   872.717786   980.936368
17  2432.0   911.035610  1022.861668
18  2560.0   945.716988  1079.441293
19  2688.0   974.622044  1094.494178
20  2816.0  1003.573511  1128.897335
21  2944.0  1014.136587  1138.465497
22  3072.0  1039.942506  1160.603599
23  3200.0  1062.350844  1174.501630
24  3328.0  1079.629479  1191.352831
25  3456.0  1100.151712  1196.422923
26  3584.0  1121.695073  1211.632234
27  3712.0  1137.882278  1219.189423
28  3840.0  1151.531540  1227.056391
29  3968.0  1170.870159  1236.428301
30  4096.0  1179.619715  1266.620583
31  4224.0  1100.824721  1244.839909
32  4352.0  1124.675334  1250.632962
33  4480.0  1134.676206  1253.498138
34  4608.0  1150.053849  1262.409900
35  4736.0  1163.205998  1266.163285
36  4864.0  1177.009500  1269.583782
37  4992.0  1187.487130  1276.067979
38  5120.0  1201.306005  1282.862641
39  5248.0  1207.741035  1285.616210
40  5376.0  1216.721817  1286.584930
41  5504.0  1223.703900  1292.169416
42  5632.0  1232.054262  1303.203463
43  5760.0  1238.334759  1299.111280
44  5888.0  1240.377828  1308.509059
45  6016.0  1248.824757  1307.595472
46  6144.0  1255.985111  1330.434079
47  6272.0  1258.592664  1320.042868
48  6400.0  1261.860760  1328.669423
49  6528.0  1271.028087  1327.597837
50  6656.0  1277.247886  1331.193133
51  6784.0  1275.992115  1336.171921
52  6912.0  1280.765304  1339.215510
53  7040.0  1284.519897  1341.135006
54  7168.0  1291.591332  1344.267524
55  7296.0  1289.995272  1341.680330
56  7424.0  1296.975234  1344.327433
57  7552.0  1295.822974  1340.461218

'''
