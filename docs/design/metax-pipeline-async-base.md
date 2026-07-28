# PipelineAsyncBase 设计

## 目标

`PipelineAsyncBase` 消费 `SplitTensorMap` 标记的 load/dot 循环，将
同步全局 load 改写为：

```text
ttg.async_copy_global_to_local
ttg.gvm_arrive
ttg.barrier_shared
ttg.local_load
tt.dot
```

并通过 Triton `PipelineExpander` 把 global-to-shared 搬运、
shared-to-register load 和 dot 交错。MetaX 不插入
`commit_group`，而是用 `gvm_arrive` 的剩余 LDG 数量表达等待点。

## 参数语义

| 参数 | 含义 |
| --- | --- |
| `numStages` / Python `num_stages` | K 方向 outer buffer 数 |
| `stage_m`, `stage_n` | `SplitTensorMap` 写入的 M/N inner 切分数 |
| `isFullStage`, `mixed` | 保留在 pass 接口中；当前新调度逻辑未据此分支 |

选择路径的规则是：

| 配置 | 行为 |
| --- | --- |
| `numStages=1`, `stage_m=stage_n=1` | 跳过 pipeline |
| `numStages=1`, 任一 inner stage 大于 1 | inner prefetch，单份 shared memory |
| `numStages>=2` | outer K ring buffer pipeline |

因此 inner stage 和 outer stage 是两个正交概念。前者切 M/N 并在一个 K
迭代内复用 shared memory，后者在多个 K 迭代间轮转 shared buffer。

## 输入契约

pass 只收集直接位于目标 `scf.for` 中、且带
`metax.split_tensor_map.operand` 的 `tt.load`。每个 load 必须满足
`canBeAsyncLoad`。

load 到 dot operand 之间允许以下单 use 链：

```text
tt.load
  -> [ttg.local_alloc | tt.trans]*
  -> [ttg.convert_layout | ttg.local_load]
```

链尾类型必须使用 `ttg.dot_op` encoding。无法识别的 load 或无法映射到
inner stage 的 dot 会让该循环转换失败。

## Load lowering

每个目标 load 独立完成以下改写：

1. 通过 `getSharedEncoding(load)` 选择 shared layout。
2. 创建 shared allocation。
3. 创建写入 view 和 `ttg.async_copy_global_to_local`。
4. 创建读取 view 和 `ttg.local_load`。
5. 用 local-load 结果替换 dot-operand conversion，并删除失效的原
   load/conversion 链。

inner prefetch 的 allocation distance 为 1。每个切分 tile 只有一个
shared slice，所有 A/B tile 的总容量等于一份完整 A/B tile。

outer pipeline 的 allocation distance 为 `numStages`。buffer index 为：

```text
iteration = (iv - lowerBound) / step
buffer = iteration % numStages
```

因此各 K 迭代在 shared ring 中循环复用槽位。

## Dot 就绪 stage

pass 先按 `(inner_stage, A-before-B)` 排序 load，再反向遍历每条 dot
前两个 operand 的依赖。遇到 `ttg.local_load` 时记录其
`inner_stage`，一条 dot 的 ready stage 是两个输入 stage 的最大值。

该映射保证 dot 只会排在其所有 shared-to-register load 之后。若循环中
存在无法关联到已标记 local load 的 dot，pipeline 不会继续生成。

## GVM arrive 和 barrier

`getGVMNumberPerOp(load)` 根据 pointer layout、AxisInfo contiguity、
元素位宽和每线程重复次数估算一条 async copy 最终产生的 MetaX LDG
数量。所有 load 的数量之和记为 `totalGVM`。

inner prefetch 当前对每个 inner stage 使用：

```text
remainingGVM = totalGVM / 2
gvm_arrive(remainingGVM)
barrier_shared
local_load
```

outer pipeline 先统计每个 inner stage 的 `gvmByInnerStage`。stage `s`
的等待阈值为：

```text
futureIterationGVM = (numStages - 2) * totalGVM
currentRemainingGVM =
    totalGVM - cumulativeGVMThroughStage(s)
remainingGVM = futureIterationGVM + currentRemainingGVM
```

`gvm_arrive` 等待队列下降到该剩余数量，随后
`barrier_shared` 保证所有线程对 shared 的写入可见。barrier 只放在
`local_load` 前，不在其后重复插入。

## Inner prefetch 调度

此路径在 `numStages=1` 且 inner stage 大于 1 时启用，PipelineExpander
使用两个逻辑 stage：

```text
pipeline stage 0: async copy / 下一 K 迭代的 register prefetch
pipeline stage 1: 当前 K 迭代的 local load / dot
```

算法步骤：

1. 最多选择前两个 inner stage 作为 register-prefetch stage。
2. 它们的 `local_load` 结果跨 pipeline stage，成为 `scf.for`
   loop-carried dot operand。
3. 对剩余 inner stage，按
   `arrive -> barrier -> local_load -> dot` 消费当前 shared slice。
4. 一个 slice 被消费后才安排其下一次 async copy。
5. wraparound 时先同步第一个 prefetched slice，再填充最后一个 slice，
   并为下一 K 迭代准备 prefetched register 值。

结果是 async copy、local load 和 dot 混插，同时 shared memory 只保留
一份。提前执行 local load 也缩短了 shared slice 的生命周期，使其可以
更早复用。

以 `stage_m=stage_n=4` 为例，`SplitTensorMap` 生成 4 个 A load、
4 个 B load 和 16 个 dot。经过 pipeline 后，前两组 A/B local-load
结果作为循环参数传递，其余组在循环体中边消费边回填。

## Outer ring 调度

`numStages>=2` 时，pass 不要求 M/N 实际切分；`(1,1)` 标记即可启用。
PipelineExpander 的调度为：

```text
stage 0:         为更远 K 迭代发起 async copy
stage N - 1:     arrive + barrier + local_load
stage N:         dot
```

yield 侧的 pointer increment 依赖会提前到 stage 1，确保 stage 0 的
copy 使用下一迭代的 loop-carried pointer。load、copy、dot 分组后由
PipelineExpander 生成 prologue、steady state 和 epilogue，从而让不同
K 迭代的访存与计算重叠。

## Predicate 和清理

动态循环由 `supportDynamicLoops=true` 支持，epilogue 不 peel。
PipelineExpander 为普通 op 生成 mask；`gvm_arrive` 和
`barrier_shared` 不包进 mask，避免线程间同步分歧。

转换完成后执行：

```text
resolveMaskOp(module)
removePipeliningAttributes(module)
```

## 限制

- 依赖 `SplitTensorMap` 的属性契约，不能独立匹配任意 load/dot。
- 目标 load 必须可异步化，且从 load 到 dot operand 必须是单 use 的
  受支持链。
- 循环中的每条 dot 都必须能映射到 inner stage。
- inner 路径的 `totalGVM / 2` 假定 A/B 搬运能够按当前成对调度模型
  对齐；新增不对称 tile 时需要重新审查等待阈值。
- 当前实现为每个切分 load 单独分配 shared slice，未做跨 load 的
  allocation 合并。
- `SplitTensorMap` 的 layout 可切分限制同样适用于本 pass；shape
  可整除不代表物理 layout 一定支持。

## 验证

`python/tutorials/03-matrix-multiplication.py` 提供两类验证：

1. `inner_stages=(4,4), num_stages=1, block=128` 与
   `torch.matmul` 的精度比较。
2. `inner_stages=(1,1), num_stages=1/2, block=64` 的无 pipeline 与
   outer pipeline 性能及精度比较。
