#ifndef TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {

namespace METAX {

using IndexCacheKeyT = std::pair<Attribute, mlir::triton::MemDescType>;

struct CacheKeyDenseMapInfo {
  // EmptyKey and TombstoneKey and HashValue are both implemented in the
  // DenseMapInfo<>template class. LLVM provides specialized versions of
  // DenseMapInfo<> for common types, such as pointer types and pointers. For
  // the KeyType we define ourselves, we need to provide DenseMapInfo
  // implementation for the KeyType. getEmptyKey(): Get null keys.By default,
  // all elements will be set to null keys during initialization to indicate
  // that the bucket is available. getTombstoneKey(): Get Tombstone Key Due to
  // possible hash conflicts, when one hash value corresponds to multiple keys,
  // the key value cannot be set to an empty key when deleting the middle
  // element, otherwise the search cannot continue.
  //  Therefore, a special marker (tombstone) is used.
  // getHashValue(): Perform hash operation on the input key
  static IndexCacheKeyT getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return std::make_pair(
        mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
        mlir::triton::MemDescType{});
  }
  static IndexCacheKeyT getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    auto tombstone =
        llvm::DenseMapInfo<mlir::triton::MemDescType>::getTombstoneKey();
    return std::make_pair(
        mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
        tombstone);
  }
  static unsigned getHashValue(IndexCacheKeyT key) {
    auto shape = key.second.getShape();
    return llvm::hash_combine(mlir::hash_value(key.first),
                              mlir::hash_value(key.second));
  }
  static bool isEqual(IndexCacheKeyT LHS, IndexCacheKeyT RHS) {
    return LHS == RHS;
  }
};

struct IndexCacheInfoSm {
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      *baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      *indexCache;
  OpBuilder::InsertPoint *indexInsertPoint;
};

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

// void populateClusterOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
//                                       RewritePatternSet &patterns,
//                                       PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit,
    IndexCacheInfoSm indexCacheInfoSm, bool enSmIdxCache, bool enSmIndexOpt);

void populateSyncOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  const TargetInfo &targetInfo,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateConvertLayoutOpToLLVMOptimizedPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit);

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

void populateViewOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  const TargetInfo &targetInfo,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  PatternBenefit benefit);

} // namespace METAX
} // namespace triton
} // namespace mlir

#endif
