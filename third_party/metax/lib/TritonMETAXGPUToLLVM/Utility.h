#ifndef TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_UTILITY_H

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "./Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define barSync(rewriter, op, bar, numThreads)                                 \
  do {                                                                         \
    ::mlir::triton::PTXBuilder ptxBuilder;                                     \
    auto &barSyncOp = *ptxBuilder.create<>("bar.sync");                        \
    barSyncOp(ptxBuilder.newConstantOperand(bar),                              \
              ptxBuilder.newConstantOperand(numThreads));                      \
    auto voidTy = void_ty(op->getContext());                                   \
    ptxBuilder.launch(rewriter, op->getLoc(), voidTy);                         \
  } while (0)

#define load_dsmem(...)                                                        \
  ::mlir::LLVM::METAX::createLoadDSmem(loc, rewriter, __VA_ARGS__)
#define store_dsmem(...)                                                       \
  ::mlir::LLVM::METAX::createStoreDSmem(loc, rewriter, __VA_ARGS__)

namespace mlir {
namespace LLVM {

namespace METAX {

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr);
Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i, Value tid, Operation *op);
Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i, Value laneId, Operation *op);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i, Operation *op);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i, Operation *op);
Value permute(Location loc, ConversionPatternRewriter &rewriter, Value a,
              Value b, Value mask);

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis);

/// Usage of macro load_dsmem
/// (1) load_dsmem(addr, ctaId)
/// (2) load_dsmem(addr, ctaId, vec)
Value createLoadDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Type elemTy);
SmallVector<Value> createLoadDSmem(Location loc, PatternRewriter &rewriter,
                                   Value addr, Value ctaId, unsigned vec,
                                   Type elemTy);

/// Usage of macro store_dsmem
/// (1) store_dsmem(addr, ctaId, value, pred)
/// (2) store_dsmem(addr, ctaId, value)
/// (3) store_dsmem(addr, ctaId, values, pred)
/// (4) store_dsmem(addr, ctaId, values)
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value, Value pred);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values, Value pred);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values);

// /// Create a predicate with just single active thread.
// Value createElectPredicate(Location loc, PatternRewriter &rewriter);
Type getFunctionType(Type resultType, const ValueRange &operands);

LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                   Operation *op, StringRef funcName,
                                   Type funcType);

__attribute__((optnone)) __attribute__((noinline)) Value createBuiltinFunc(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    StringRef funcName, Type resultType, const ValueRange &argValues);
} // namespace METAX

#ifdef USE_MACA

Type getFunctionType(Type resultType, const ValueRange &operands);

template <typename T>
LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter, T op,
                                   StringRef funcName, Type funcType) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  mlir::OpBuilder b(op->template getParentOfType<LLVMFuncOp>());
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  return ret;
}

template <typename T>
__attribute__((optnone)) __attribute__((noinline)) Value createBuiltinFunc(
    ConversionPatternRewriter &rewriter, Location loc, T op, StringRef funcName,
    Type resultType, const ValueRange &argValues) {
  Type funcType = getFunctionType(resultType, argValues);
  auto funcOp = appendOrGetFuncOp<T>(rewriter, op, funcName, funcType);
  return rewriter.create<LLVM::CallOp>(loc, funcOp, argValues).getResult();
}

template <typename T>
DenseMap<unsigned, Value> getNoSwizzledSharedPtrs(
    Location loc, ConversionPatternRewriter &rewriter,
    const TargetInfoBase &target, T op, unsigned inVec, RankedTensorType srcTy,
    triton::gpu::SharedEncodingAttr resSharedLayout, Type resElemTy,
    SharedMemoryObject smemObj, SmallVectorImpl<Value> &offsetVals,
    SmallVectorImpl<Value> &srcStrides) {
  unsigned maxPhase = resSharedLayout.getMaxPhase();
  if (maxPhase == 1) {
    return getSwizzledSharedPtrs(loc, target, inVec, srcTy, resSharedLayout,
                                 resElemTy, smemObj, rewriter, offsetVals,
                                 smemObj.strides);
  }

#ifndef USE_MACA_OPAQUE_PTR
  auto dstPtrTy = ptr_ty(resElemTy, 3);
  ;
#else
  auto dstPtrTy = ptr_ty(op->getContext(), 3);
#endif
  auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
  Value dstPtrBase = gep(dstPtrTy, resElemTy, smemObj.base, dstOffset);

  auto srcEncoding = srcTy.getEncoding();
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);

  unsigned outVec = resSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  // order
  auto inOrder = triton::gpu::getOrder(srcEncoding);
  auto outOrder = triton::gpu::getOrder(resSharedLayout);

  // tensor indices held by the current thread, as LLVM values
  auto srcIndices = emitIndices(loc, rewriter, target, srcEncoding, srcTy,
                                /*withCTAOffset=*/false);
  // return values
  DenseMap<unsigned, Value> ret;

  for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
    // extract multi dimensional index for current element
    auto idx = srcIndices[elemIdx];
    Value idxCol = idx[outOrder[0]]; // contiguous dimension
    Value idxRow = idx[outOrder[1]]; // discontiguous dimension
    Value strideCol = srcStrides[outOrder[0]];
    Value strideRow = srcStrides[outOrder[1]];
    Value offset = add(mul(idxRow, strideRow), mul(idxCol, strideCol));
    std::string rfl_name = "llvm.mxc.rfl";
    ValueRange rflValues(offset);
    Value offset_st = mlir::LLVM::createBuiltinFunc<T>(
        rewriter, loc, op, rfl_name, i32_ty, rflValues);
    Value currPtr = gep(dstPtrTy, resElemTy, dstPtrBase, offset_st);
    ret[elemIdx] = currPtr;
  }
  return ret;
}

#endif

} // namespace LLVM

} // namespace mlir

namespace maca {
bool get_triton_disable_fp32_to_bf16_high_precsion();

void appendIntrinsicModifer(std::string &str, int vec, Type elemType);

} // namespace maca

#endif
