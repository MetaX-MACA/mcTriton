# SplitTensorMap 设计

## 目标

`SplitTensorMap` 将一个 TN `tt.dot` 在 M、N 方向切成多个更小的
load/dot。切分后的 load 可以独立搬运和同步，为后续
`PipelineAsyncBase` 的 inner-stage 共享内存复用提供调度粒度。

Python 编译参数与 pass 参数的对应关系如下：

| Python 参数 | Pass 参数 | 含义 |
| --- | --- | --- |
| `inner_stages[0]` | `innerStageM` | A 和 C 在 M 方向的切分数 |
| `inner_stages[1]` | `innerStageN` | B 和 C 在 N 方向的切分数 |

`num_stages` 不控制本 pass。它在后续 `PipelineAsyncBase` 中表示 K
方向的 outer buffer 数。

## Pass 位置

MetaX `cpasync` pipeline 的关键顺序是：

```text
AddPtrOpt
RemoveLayoutConversions
SplitTensorMap
PipelineAsyncTT
PipelineAsyncBase
```

`SplitTensorMap` 先建立可独立调度的 tile 和属性契约；
`PipelineAsyncBase` 再消费这些属性。`PipelineAsyncTT` 位于两者之间，
优先处理适合 TT pipeline 的循环。

## 匹配条件

当前实现只处理满足以下条件的 `tt.dot`：

1. `tt.dot` 直接位于 `scf.for` 中，且该循环尚未带
   `metax.split_tensor_map` 属性。
2. A、B 都是二维 tensor，C 的 shape 与矩阵乘法一致：
   A 为 `[M, K]`，B 为 `[K, N]`，C 为 `[M, N]`。
3. 两个输入均符合 `tt.load -> ttg.convert_layout -> tt.dot`。
   `tt.load` 必须与 dot 位于同一个 `scf.for`。
4. `stageM > 0`、`stageN > 0`，并且
   `M % stageM == 0`、`N % stageN == 0`。

当任一 stage 配置为 0 时，pass 使用
`matchMNStage({M, N, K, numWarps}, Layout::TN, dtype)` 查询 MetaX
stage 表。查表失败返回 `(0, 0)`，该 dot 不会被切分。

## 切分算法

令：

```text
subM = M / stageM
subN = N / stageN
```

输入切分为：

```text
A[M, K] -> stageM 个 A_i[subM, K]
B[K, N] -> stageN 个 B_j[K, subN]
```

对 pointer、mask 和 tensor 类型的 `other`，pass 调用
`calExtractTensorIdx` 计算 CTA/element 索引，并生成
`ttg.extract_tensor`。随后复制原 `tt.load` 的 boundary check、padding、
cache、evict、volatile 和 contiguity 属性，生成子 load。

每个子 load 经过使用原 dot-operand encoding 的
`ttg.convert_layout`。原 dot 被替换为：

```text
for i in [0, stageM):
  for j in [0, stageN):
    C_ij = extract_tensor(C, [i, j])
    R_ij = dot(A_i, B_j, C_ij)
    result = insert_tensor(result, R_ij, [i, j])
```

因此一条 dot 会变成 `stageM * stageN` 条 dot。A 只切 M，B 只切 N，
K 方向仍由原 `scf.for` 迭代。

## 属性契约

切分后的 load 带有：

```text
metax.split_tensor_map.operand = "A" | "B"
metax.split_tensor_map.inner_stage = i | j
```

循环带有：

```text
metax.split_tensor_map
metax.split_tensor_map.stage_m = stageM
metax.split_tensor_map.stage_n = stageN
```

`PipelineAsyncBase` 依赖这些属性识别 load、按 inner stage 分组，并把
dot 关联到其输入就绪的 stage。修改属性名或 stage 编号规则时必须同步
更新两个 pass。

## 1x1 特例

`inner_stages=(1, 1)` 不改变 tensor 和 dot。pass 只标记原 A/B load
及其循环。这样 `num_stages >= 2` 时，`PipelineAsyncBase` 可以在不做
M/N 切分的情况下建立 K 方向 outer pipeline。

当 `inner_stages=(1, 1)` 且 `num_stages=1` 时，
`PipelineAsyncBase` 主动跳过该循环，相当于关闭异步 pipeline。

## Layout 约束和限制

shape 可整除只是必要条件。`ttg.extract_tensor` 还要求源 layout 的
CTA/thread/element 分布能够表示子 tensor。`calExtractTensorIdx` 根据
`MACAMmaEncodingAttr`、`BlockedEncodingAttr` 或
`DotOperandEncodingAttr` 的物理分布计算索引。

例如当前 TN tutorial 的 blocked layout 中：

- `block=128x128x128, inner_stages=(4,4)` 可切成
  `A_i[32,128]` 和 `B_j[128,32]`。
- `block=64x64x64, inner_stages=(4,4)` 会得到逻辑上的 16 元素边，
  但该 layout 无法表示对应子 tensor，类型推导会失败。

因此新增配置时必须实际编译生成 TTGIR，不能只检查 shape 整除。
当前 pass 尚未在创建 op 前验证 `calExtractTensorIdx` 的结果；不支持的
layout 可能表现为 MLIR verifier 错误，而不是静默跳过。

当前实现还假定一个待处理循环只有一组目标 dot。第一条 dot 成功后循环
会被标记，后续 dot 不再重复切分。

## 验证

`python/tutorials/03-matrix-multiplication.py` 包含
`inner_stages=(4,4), num_stages=1, block=128` 的精度测例。它覆盖
128 到 4096 的 2 次幂方阵，并与 `torch.matmul` 比较。
