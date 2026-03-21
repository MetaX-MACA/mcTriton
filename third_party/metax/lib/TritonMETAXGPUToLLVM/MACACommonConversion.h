#ifndef TRITON_MACA_COMMON_H
#define TRITON_MACA_COMMON_H

#include "Utility.h"

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

// Helper for conversion of DotOp with MACA mma
enum class TensorCoreType : uint8_t {
  // floating-point tensor core instr
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_FP32_FP32_FP32,
  FP32_TF32_TF32_FP32,
  FP64_FP64_FP64_FP64,
  INT32_INT8_INT8_INT32,
  INT32_INT8_INT8_INT32_1, // for computeCapability >= 86 or 89
  FP32_FP8_FP8_FP32,
  FP32_BF8_BF8_FP32,
  // TODO(liuyuxin) More tensor core type to be suppored!
  NOT_APPLICABLE,
};

#endif
