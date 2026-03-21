#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H

#include <set>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#ifdef USE_MACA
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"
#endif

#define DEBUG_TYPE "ttgpu_to_llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define inttofloat(...) rewriter.create<LLVM::SIToFPOp>(loc, __VA_ARGS__)
#define inttoptr(...) rewriter.create<LLVM::IntToPtrOp>(loc, __VA_ARGS__)
#define ptrtoint(...) rewriter.create<LLVM::PtrToIntOp>(loc, __VA_ARGS__)
#define zext(...) rewriter.create<LLVM::ZExtOp>(loc, __VA_ARGS__)
#define sext(...) rewriter.create<LLVM::SExtOp>(loc, __VA_ARGS__)
#define fpext(...) rewriter.create<LLVM::FPExtOp>(loc, __VA_ARGS__)
#define trunc(...) rewriter.create<LLVM::TruncOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<LLVM::UDivOp>(loc, __VA_ARGS__)
#define urem(...) rewriter.create<LLVM::URemOp>(loc, __VA_ARGS__)
#define add(...) rewriter.create<LLVM::AddOp>(loc, __VA_ARGS__)
#define sub(...) rewriter.create<LLVM::SubOp>(loc, __VA_ARGS__)
#define fadd(...) rewriter.create<LLVM::FAddOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<LLVM::MulOp>(loc, __VA_ARGS__)
#define fmul(...) rewriter.create<LLVM::FMulOp>(loc, __VA_ARGS__)
#define smax(...) rewriter.create<LLVM::SMaxOp>(loc, __VA_ARGS__)
#define umax(...) rewriter.create<LLVM::UMaxOp>(loc, __VA_ARGS__)
#define fmax(...) rewriter.create<LLVM::MaxNumOp>(loc, __VA_ARGS__)
#define smin(...) rewriter.create<LLVM::SMinOp>(loc, __VA_ARGS__)
#define umin(...) rewriter.create<LLVM::UMinOp>(loc, __VA_ARGS__)
#define fmin(...) rewriter.create<LLVM::MinNumOp>(loc, __VA_ARGS__)
#define shl(...) rewriter.create<LLVM::ShlOp>(loc, __VA_ARGS__)
#define lshr(...) rewriter.create<LLVM::LShrOp>(loc, __VA_ARGS__)
#define and_(...) rewriter.create<LLVM::AndOp>(loc, __VA_ARGS__)
#define xor_(...) rewriter.create<LLVM::XOrOp>(loc, __VA_ARGS__)
#define or_(...) rewriter.create<LLVM::OrOp>(loc, __VA_ARGS__)
#define bitcast(val__, type__)                                                 \
  rewriter.create<LLVM::BitcastOp>(loc, type__, val__)
#define addrspacecast(...)                                                     \
  rewriter.create<LLVM::AddrSpaceCastOp>(loc, __VA_ARGS__)
#define gep(...) rewriter.create<LLVM::GEPOp>(loc, __VA_ARGS__)
#define ptr_ty(...) LLVM::LLVMPointerType::get(__VA_ARGS__)
#define insert_val(...) rewriter.create<LLVM::InsertValueOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<LLVM::ExtractValueOp>(loc, __VA_ARGS__)
#define insert_element(...)                                                    \
  rewriter.create<LLVM::InsertElementOp>(loc, __VA_ARGS__)
#define extract_element(...)                                                   \
  rewriter.create<LLVM::ExtractElementOp>(loc, __VA_ARGS__)
#define load(...) rewriter.create<LLVM::LoadOp>(loc, __VA_ARGS__)
#define store(...) rewriter.create<LLVM::StoreOp>(loc, __VA_ARGS__)
#define fcmp_ogt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::ogt, lhs, rhs)
#define fcmp_olt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::olt, lhs, rhs)
#define fcmp_eq(lhs, rhs)                                                      \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::oeq, lhs, rhs)
#define icmp_eq(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, __VA_ARGS__)
#define icmp_ne(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, __VA_ARGS__)
#define icmp_slt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, __VA_ARGS__)
#define icmp_sle(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, __VA_ARGS__)
#define icmp_sgt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, __VA_ARGS__)
#define icmp_sge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, __VA_ARGS__)
#define icmp_ult(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, __VA_ARGS__)
#define icmp_ule(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ule, __VA_ARGS__)
#define icmp_ugt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, __VA_ARGS__)
#define icmp_uge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge, __VA_ARGS__)
#define select(...) rewriter.create<LLVM::SelectOp>(loc, __VA_ARGS__)
#define address_of(...) rewriter.create<LLVM::AddressOfOp>(loc, __VA_ARGS__)
#define barrier() rewriter.create<mlir::gpu::BarrierOp>(loc)
#define undef(...) rewriter.create<LLVM::UndefOp>(loc, __VA_ARGS__)
#define null(...) rewriter.create<LLVM::ZeroOp>(loc, __VA_ARGS__)
#define call(...) rewriter.create<LLVM::CallOp>(loc, __VA_ARGS__)

// Types
#define int_ty(width) rewriter.getIntegerType(width)
#define i64_ty rewriter.getIntegerType(64)
#define i32_ty rewriter.getIntegerType(32)
#define i16_ty rewriter.getIntegerType(16)
#define i32_ty rewriter.getIntegerType(32)
#define i64_ty rewriter.getIntegerType(64)
#define ui32_ty rewriter.getIntegerType(32, false)
#define ui64_ty rewriter.getIntegerType(64, false)
#define f16_ty rewriter.getF16Type()
#define bf16_ty rewriter.getBF16Type()
#define i8_ty rewriter.getIntegerType(8)
#define i1_ty rewriter.getI1Type()
#define f32_ty rewriter.getF32Type()
#define f64_ty rewriter.getF64Type()
#define vec_ty(type, num) VectorType::get(num, type)
#define void_ty(ctx) LLVM::LLVMVoidType::get(ctx)
#define struct_ty(...) LLVM::LLVMStructType::getLiteral(ctx, __VA_ARGS__)
#define array_ty(elemTy, count) LLVM::LLVMArrayType::get(elemTy, count)

// Constants
#define f16_val(...) LLVM::createConstantF16(loc, rewriter, __VA_ARGS__)
#define f32_val(...) LLVM::createConstantF32(loc, rewriter, __VA_ARGS__)
#define f64_val(...) LLVM::createConstantF64(loc, rewriter, __VA_ARGS__)
#define i32_val(...) LLVM::createConstantI32(loc, rewriter, __VA_ARGS__)
#define i64_val(...) LLVM::createConstantI64(loc, rewriter, __VA_ARGS__)
#define int_val(width, val)                                                    \
  LLVM::createLLVMIntegerConstant(rewriter, loc, width, val)
#define i8_val(val) int_val(8, val)
#define i1_val(val) int_val(1, val)
#define i16_val(val) int_val(16, val)
#define tid_val() getThreadId(rewriter, loc)

// Attributes
#define i32_arr_attr(...) rewriter.getI32ArrayAttr({__VA_ARGS__})
#define i64_arr_attr(...) rewriter.getI64ArrayAttr({__VA_ARGS__})
#define str_attr(str) ::mlir::StringAttr::get(ctx, (str))

#define debug_print (std::getenv("TRITON_ENABLE_SMEM_OFFSET_PRINT")==nullptr) ? (llvm::nulls()) : (llvm::errs())

namespace mlir {
namespace triton {

// Delinearize supposing order is [0, 1, .. , n]
template <typename T>
llvm::SmallVector<T> getMultiDimIndexImpl(T linearIndex,
                                          llvm::ArrayRef<T> shape) {
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearRemain = linearIndex;
  llvm::SmallVector<T> multiDimIndex(rank);
  for (int i = rank - 1; i >= 0; --i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
llvm::SmallVector<T> getMultiDimIndex(T linearIndex, llvm::ArrayRef<T> shape,
                                      llvm::ArrayRef<unsigned> order) {
  size_t rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  auto reorderedMultiDim = getMultiDimIndexImpl<T>(linearIndex, reordered);
  llvm::SmallVector<T> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

// Linearize supposing order is [0, 1, .. , n]
template <typename T>
T getLinearIndexImpl(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
T getLinearIndex(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape,
                 llvm::ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getLinearIndexImpl<T>(applyPermutation(multiDimIndex, order),
                               applyPermutation(shape, order));
}

namespace gpu {
Type getFunctionType(Type resultType, ValueRange operands);

LLVM::LLVMFuncOp appendOrGetExternFuncOp(ConversionPatternRewriter &rewriter,
                                         Operation *op, StringRef funcName,
                                         Type funcType, StringRef libname = "",
                                         StringRef libpath = "");
} // namespace gpu

} // namespace triton

namespace LLVM {
using namespace mlir::triton;

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v);

/// Create a 64-bit integer constant.
Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);

/// Create a 16-bit float constant.
Value createConstantF16(Location loc, OpBuilder &rewriter, float v);

/// Create a 32-bit float constant.
Value createConstantF32(Location loc, OpBuilder &rewriter, float v);

/// Create a 64-bit float constant.
Value createConstantF64(Location loc, OpBuilder &rewriter, double v);

/// Create NaN constant of specified type.
Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type);

/// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          const TypeConverter *converter, int64_t value);

/// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value);

LLVM::CallOp createLLVMCallOp(OpBuilder &builder, Location loc,
                              LLVMFuncOp funcOp, ValueRange args);

/// Helper function to get strides from a given shape and its order
SmallVector<Value> getStridesFromShapeAndOrder(ArrayRef<int64_t> shape,
                                               ArrayRef<unsigned> order,
                                               Location loc,
                                               RewriterBase &rewriter);
struct SharedMemoryObject {
  Value base; // i32 ptr. The start address of the shared memory object after
              // the initial allocation or the last slicing operation.
  Type baseElemType;
  // We need to store strides as Values, not integers, because the
  // extract_slice instruction can take a slice at arbitrary offsets.
  // Take $a[16:32, 16:32] as an example; though we know the stride of $a[0] is
  // 32, we need to let the instruction that uses $a be aware of that.
  // Otherwise, when we use $a, we only know that the shape of $a is 16x16. If
  // we store strides into an attribute array of integers, the information
  // cannot pass through block argument assignment because attributes are
  // associated with operations, not Values.
  // TODO(Keren): We may need to figure out a way to store strides as integers
  // if we want to support more optimizations.
  SmallVector<Value>
      strides; // i32 int. The strides of the shared memory object.
  SmallVector<Value> offsets; // i32 int.
  // Offsets are applied at the last slicing operation.
  // We can use offsets to recover the previous base.
  // The offsets are zero at the initial allocation.
#ifdef USE_MACA
  Value static_offset;
  SharedMemoryObject(Value base, Type baseElemType, ArrayRef<Value> strides,
                     ArrayRef<Value> offsets, Value static_offset)
      : base(base), baseElemType(baseElemType), strides(strides.begin(), strides.end()),
        offsets(offsets.begin(), offsets.end()), static_offset(static_offset) {}
#endif

  SharedMemoryObject(Value base, Type baseElemType, ArrayRef<Value> strides,
                     ArrayRef<Value> offsets)
      : base(base), baseElemType(baseElemType),
        strides(strides.begin(), strides.end()),
        offsets(offsets.begin(), offsets.end()) {}

  SharedMemoryObject(Value base, Type baseElemType, ArrayRef<int64_t> shape,
                     ArrayRef<unsigned> order, Location loc,
                     RewriterBase &rewriter)
      : base(base), baseElemType(baseElemType) {
    strides = getStridesFromShapeAndOrder(shape, order, loc, rewriter);
    offsets.append(order.size(), i32_val(0));
  }

  SmallVector<Value> getStrides() const { return strides; }
  SmallVector<Value> getOffsets() const { return offsets; }
  Value getBase() const { return base; }
  Type getBaseElemType() const { return baseElemType; }

#ifdef USE_MACA
  SmallVector<Value> getElems(Location loc,
                              ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> elems;
    elems.push_back(base);
    elems.append(strides.begin(), strides.end());
    elems.append(offsets.begin(), offsets.end());
    if (static_offset) {
      elems.push_back(static_offset);
    } else {
      elems.push_back(i32_val(0));
    }
#else
  SmallVector<Value> getElems() const {
    SmallVector<Value> elems;
    elems.push_back(base);
    elems.append(strides.begin(), strides.end());
    elems.append(offsets.begin(), offsets.end());
#endif
    return elems;
  }

  SmallVector<Type> getTypes() const {
    SmallVector<Type> types;
    types.push_back(base.getType());
    types.append(strides.size(), IntegerType::get(base.getContext(), 32));
    types.append(offsets.size(), IntegerType::get(base.getContext(), 32));
#ifdef USE_MACA
    types.push_back(IntegerType::get(base.getContext(), 32));
#endif
    return types;
  }

  Value getCSwizzleOffset(int order) const {
    assert(order >= 0 && order < strides.size());
    return offsets[order];
  }

  Value getBaseBeforeSlice(int order, Location loc,
                           ConversionPatternRewriter &rewriter) const {
    Value cSwizzleOffset = getCSwizzleOffset(order);
    Value offset = sub(i32_val(0), cSwizzleOffset);
    Type type = base.getType();
    return gep(type, baseElemType, base, offset);
  }
};

SharedMemoryObject
getSharedMemoryObjectFromStruct(Location loc, Value llvmStruct, Type elemTy,
                                ConversionPatternRewriter &rewriter);

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order);

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape);

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape);

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape,
                ArrayRef<unsigned> order);

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape);

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content);

// Given an elemId which represents the index of an element from the list of
// elements that are in the thread's registers (i.e. total of
// numel(sizePerThread)), it calculates the multi dim offset of the element in
// the smem buffer. Recall that the smem buffer will only store a single replica
// when converting distributed to distributed layout. Also, a replica is the
// smallest CTA tile that is common between input and output layouts.
SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const TargetInfoBase &targetInfo,
                                     unsigned elemId, RankedTensorType type,
                                     ArrayRef<unsigned> multiDimCTAInRepId,
                                     ArrayRef<unsigned> shapePerCTATile);

// Given a multiDimOffset, this function wraps around each dimension to be
// within shape.
SmallVector<Value> getWrappedMultiDimOffset(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Value> multiDimOffset, ArrayRef<unsigned> shape,
    SmallVector<unsigned> shapePerCTATile, SmallVector<int64_t> shapePerCTA);

inline bool isKernel(FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

inline Value getStackPointer(PatternRewriter &rewriter,
                             FunctionOpInterface funcOp) {
  auto mod = funcOp->getParentOfType<ModuleOp>();
  LLVM::GlobalOp globalBase = nullptr;
  mod.walk([&](LLVM::GlobalOp op) {
    if (op.getSymName() == "global_smem")
      globalBase = op;
  });
  assert(globalBase);
  if (isKernel(funcOp))
    return rewriter.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalBase);
  else
    return funcOp.getArgument(funcOp.getNumArguments() - 1);
}

inline Value getSharedMemoryBase(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Operation *op) {
#ifndef USE_MACA_OPAQUE_PTR
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), i8_ty, 3);
#else
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
#endif
  FunctionOpInterface func =
      op->template getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = cast<IntegerAttr>(op->getAttr("allocation.offset"))
                      .getValue()
                      .getZExtValue();
  Value offVal = i32_val(offset);
  Value base = gep(ptrTy, i8_ty, LLVM::getStackPointer(rewriter, func), offVal);
  return base;
}


#ifdef USE_MACA
inline Value getStackPointer(PatternRewriter &rewriter,
                             FunctionOpInterface funcOp,
                             const LLVMTypeConverter *typeConverter) {
  auto mod = funcOp->getParentOfType<ModuleOp>();
  LLVM::GlobalOp globalBase = nullptr;
  mod.walk([&](LLVM::GlobalOp op) {
    if (op.getSymName() == "global_smem")
      globalBase = op;
  });
  assert(globalBase);

  OpBuilder b(mod.getBodyRegion());
  b.setInsertionPointToStart(&funcOp.getFunctionBody().front());
  auto loc = mod.getLoc();
  Value funcSmem;
  if (isKernel(funcOp)) {
    funcSmem = rewriter.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalBase);
  } else {
    funcSmem = funcOp.getArgument(funcOp.getNumArguments() - 1);
  }
  return funcSmem;
}

inline Value getSharedMemoryBase(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Operation *op, const LLVMTypeConverter *typeConverter) {
#ifndef USE_MACA_OPAQUE_PTR
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), i8_ty, 3);
#else
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
#endif
  FunctionOpInterface func =
      op->template getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = cast<IntegerAttr>(op->getAttr("allocation.offset"))
                      .getValue()
                      .getZExtValue();
  Value offVal = i32_val(offset);

  Value base = gep(ptrTy, i8_ty, bitcast(LLVM::getStackPointer(rewriter, func, typeConverter),ptrTy), offVal);
  return base;
}
#endif


} // namespace LLVM

/* ------------------------------------ */
// Returns CTA level thread idx
inline Value getThreadIdInCTA(RewriterBase &rewriter, Location loc) {
  Value tid =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);
}

// Returns CTA level thread idx.
inline Value getThreadId(RewriterBase &rewriter, Location loc) {
  Value tid = getThreadIdInCTA(rewriter, loc);
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  return tid;
}

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
using LLVM::getMultiDimIndex;
using LLVM::SharedMemoryObject;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::MACAMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

inline Value dot(RewriterBase &rewriter, Location loc, ArrayRef<Value> offsets,
                 ArrayRef<Value> strides) {
  assert(offsets.size() == strides.size());
  Value ret = i32_val(0);
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    ret = add(ret, mul(offset, stride));
  }
  return ret;
}

// -----------------------------------------------------------------------
// Blocked layout indices
// -----------------------------------------------------------------------

// "Applies" the given layout by computing layout(indices) and returning the
// resulting Values.
//
// In other words, this generates LLVM-dialect MLIR code to "run" the layout
// function.
SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(Location loc, RewriterBase &rewriter,
                  const LinearLayout &layout,
                  ArrayRef<std::pair<StringAttr, Value>> indices);

inline SmallVector<Value>
emitBaseIndexWithinCTAForBlockedLayout(Location loc, RewriterBase &rewriter,
                                       const BlockedEncodingAttr &blockedLayout,
                                       RankedTensorType type) {
  MLIRContext *ctx = rewriter.getContext();
  auto shape = type.getShape();
  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(blockedLayout));
  Value laneId = urem(threadId, warpSize);
  Value warpId = udiv(threadId, warpSize);
  auto sizePerThread = blockedLayout.getSizePerThread();
  auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
  auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
  auto order = blockedLayout.getOrder();
  auto shapePerCTA = triton::gpu::getShapePerCTA(blockedLayout, shape);
  unsigned rank = shape.size();

  // delinearize threadId to get the base index
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);
  SmallVector<Value> multiDimThreadId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, order);

  SmallVector<Value> multiDimBase(rank);
  for (unsigned k = 0; k < rank; ++k) {
    // Wrap around multiDimWarpId/multiDimThreadId in case
    // shapePerCTATile[k] > shapePerCTA[k]
    auto maxWarps =
        ceil<unsigned>(shapePerCTA[k], sizePerThread[k] * threadsPerWarp[k]);
    auto maxThreads = ceil<unsigned>(shapePerCTA[k], sizePerThread[k]);
    multiDimWarpId[k] = urem(multiDimWarpId[k], i32_val(maxWarps));
    multiDimThreadId[k] = urem(multiDimThreadId[k], i32_val(maxThreads));
    // multiDimBase[k] = (multiDimThreadId[k] +
    //                    multiDimWarpId[k] * threadsPerWarp[k]) *
    //                   sizePerThread[k];
    Value threadsPerWarpK = i32_val(threadsPerWarp[k]);
    Value sizePerThreadK = i32_val(sizePerThread[k]);
    multiDimBase[k] =
        mul(sizePerThreadK,
            add(multiDimThreadId[k], mul(multiDimWarpId[k], threadsPerWarpK)));
  }

  return multiDimBase;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForBlockedLayout(const BlockedEncodingAttr &blockedLayout,
                           RankedTensorType type) {
  auto ctx = type.getContext();
  auto shape = type.getShape();
  auto sizePerThread = blockedLayout.getSizePerThread();
  auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
  auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
  auto order = blockedLayout.getOrder();
  auto shapePerCTATile = getShapePerCTATile(blockedLayout);
  auto shapePerCTA = triton::gpu::getShapePerCTA(blockedLayout, shape);

  unsigned rank = shape.size();
  SmallVector<unsigned> tilesPerDim(rank);
  for (unsigned k = 0; k < rank; ++k)
    tilesPerDim[k] = ceil<unsigned>(shapePerCTA[k], shapePerCTATile[k]);

  unsigned elemsPerThread = triton::gpu::getTotalElemsPerThread(type);
  unsigned totalSizePerThread = product<unsigned>(sizePerThread);
  SmallVector<SmallVector<unsigned>> reorderedOffset(elemsPerThread);
  for (unsigned n = 0; n < elemsPerThread; ++n) {
    unsigned linearNanoTileId = n / totalSizePerThread;
    unsigned linearNanoTileElemId = n % totalSizePerThread;
    SmallVector<unsigned> multiDimNanoTileId =
        getMultiDimIndex<unsigned>(linearNanoTileId, tilesPerDim, order);
    SmallVector<unsigned> multiDimNanoTileElemId =
        getMultiDimIndex<unsigned>(linearNanoTileElemId, sizePerThread, order);
    for (unsigned k = 0; k < rank; ++k) {
      unsigned reorderedMultiDimId =
          (multiDimNanoTileId[k] *
               (sizePerThread[k] * threadsPerWarp[k] * warpsPerCTA[k]) +
           multiDimNanoTileElemId[k]) %
          shapePerCTA[k];

      reorderedOffset[n].push_back(reorderedMultiDimId);
    }
  }

  return reorderedOffset;
}

// -----------------------------------------------------------------------
// Mma layout indices
// -----------------------------------------------------------------------

inline SmallVector<Value>
emitBaseIndexWithinCTAForMmaLayoutV1(Location loc, RewriterBase &rewriter,
                                     const NvidiaMmaEncodingAttr &mmaLayout,
                                     RankedTensorType type) {
  auto shape = type.getShape();
  auto wpt = mmaLayout.getWarpsPerCTA();
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};
  auto [isARow, isBRow, isAVec4, isBVec4, _] =
      mmaLayout.decodeVoltaLayoutStates();

  Value thread = getThreadId(rewriter, loc);
  auto *ctx = thread.getContext();
  Value _1 = i32_val(1);
  Value _2 = i32_val(2);
  Value _4 = i32_val(4);
  Value _16 = i32_val(16);
  Value _32 = i32_val(32);
  Value _fpw0 = i32_val(fpw[0]);
  Value _fpw1 = i32_val(fpw[1]);

  // A info
  auto aRep = mmaLayout.getMMAv1Rep(0);
  auto aSpw = mmaLayout.getMMAv1ShapePerWarp(0);
  // B info
  auto bSpw = mmaLayout.getMMAv1ShapePerWarp(1);
  auto bRep = mmaLayout.getMMAv1Rep(1);

  SmallVector<int, 2> rep({aRep[0], bRep[1]});
  SmallVector<int, 2> spw({aSpw[0], bSpw[1]});
  SmallVector<unsigned, 2> shapePerCTA({spw[0] * wpt[0], spw[1] * wpt[1]});

  Value lane = urem(thread, _32);
  Value warp = udiv(thread, _32);

  Value warp0 = urem(warp, i32_val(wpt[0]));
  Value warp12 = udiv(warp, i32_val(wpt[0]));
  Value warp1 = urem(warp12, i32_val(wpt[1]));

  // warp offset
  Value offWarpM = mul(warp0, i32_val(spw[0]));
  Value offWarpN = mul(warp1, i32_val(spw[1]));
  // quad offset
  Value offQuadM = mul(udiv(and_(lane, _16), _4), _fpw0);
  Value offQuadN = mul(udiv(and_(lane, _16), _4), _fpw1);
  // pair offset
  Value offPairM = udiv(urem(lane, _16), _4);
  offPairM = urem(offPairM, _fpw0);
  offPairM = mul(offPairM, _4);
  Value offPairN = udiv(urem(lane, _16), _4);
  offPairN = udiv(offPairN, _fpw0);
  offPairN = urem(offPairN, _fpw1);
  offPairN = mul(offPairN, _4);
  offPairM = mul(offPairM, i32_val(rep[0] / 2));
  offQuadM = mul(offQuadM, i32_val(rep[0] / 2));
  offPairN = mul(offPairN, i32_val(rep[1] / 2));
  offQuadN = mul(offQuadN, i32_val(rep[1] / 2));
  // quad pair offset
  Value offLaneM = add(offPairM, offQuadM);
  Value offLaneN = add(offPairN, offQuadN);
  // a, b offset
  Value offsetAM = add(offWarpM, offLaneM);
  Value offsetBN = add(offWarpN, offLaneN);
  // m indices
  Value offsetCM = add(and_(lane, _1), offsetAM);
  // n indices
  Value offsetCN = add((and_(lane, _2)), (add(offWarpN, offPairN)));
  return {offsetCM, offsetCN};
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForMmaLayoutV1(const NvidiaMmaEncodingAttr &mmaLayout,
                         RankedTensorType type) {
  auto shape = type.getShape();

  auto [isARow, isBRow, isAVec4, isBVec4, _] =
      mmaLayout.decodeVoltaLayoutStates();

  // TODO: seems like the pattern below to get `rep`/`spw` appears quite often
  // A info
  auto aRep = mmaLayout.getMMAv1Rep(0);
  auto aSpw = mmaLayout.getMMAv1ShapePerWarp(0);
  // B info
  auto bSpw = mmaLayout.getMMAv1ShapePerWarp(1);
  auto bRep = mmaLayout.getMMAv1Rep(1);

  auto wpt = mmaLayout.getWarpsPerCTA();
  static constexpr std::array<int, 3> fpw{{2, 2, 1}};
  SmallVector<int, 2> rep({aRep[0], bRep[1]});
  SmallVector<int, 2> spw({aSpw[0], bSpw[1]});
  SmallVector<unsigned, 2> shapePerCTA({spw[0] * wpt[0], spw[1] * wpt[1]});

  SmallVector<unsigned> idxM;
  for (unsigned m = 0; m < shape[0]; m += shapePerCTA[0])
    for (unsigned mm = 0; mm < rep[0]; ++mm)
      idxM.push_back(m + mm * 2);

  SmallVector<unsigned> idxN;
  for (int n = 0; n < shape[1]; n += shapePerCTA[1]) {
    for (int nn = 0; nn < rep[1]; ++nn) {
      idxN.push_back(n + nn / 2 * 4 + (nn % 2) * 2 * fpw[1] * rep[1]);
      idxN.push_back(n + nn / 2 * 4 + (nn % 2) * 2 * fpw[1] * rep[1] + 1);
    }
  }

  SmallVector<SmallVector<unsigned>> ret;
  for (unsigned x1 : idxN) {   // N
    for (unsigned x0 : idxM) { // M
      SmallVector<unsigned> idx(2);
      idx[0] = x0; // M
      idx[1] = x1; // N
      ret.push_back(std::move(idx));
    }
  }
  return ret;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForMmaLayoutV2(const NvidiaMmaEncodingAttr &mmaLayout,
                         RankedTensorType type) {
  auto shape = type.getShape();
  auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
  SmallVector<SmallVector<unsigned>> ret;

  auto rank = shape.size();
  for (unsigned i = 0; i < shapePerCTA[rank - 2];
       i += getShapePerCTATile(mmaLayout)[rank - 2]) {
    for (unsigned j = 0; j < shapePerCTA[rank - 1];
         j += getShapePerCTATile(mmaLayout)[rank - 1]) {
      if (rank == 3) {
        ret.push_back({0, i, j});
        ret.push_back({0, i, j + 1});
        ret.push_back({0, i + 8, j});
        ret.push_back({0, i + 8, j + 1});
      } else {
        ret.push_back({i, j});
        ret.push_back({i, j + 1});
        ret.push_back({i + 8, j});
        ret.push_back({i + 8, j + 1});
      }
    }
  }
  return ret;
}

// Note that this may return a null Value for one or more dimensions.  This is
// valid only if you're going to slice off the relevant dimension.
inline SmallVector<Value>
emitBaseIndexWithinCTAForMmaLayoutV2V3(Location loc, RewriterBase &rewriter,
                                       const NvidiaMmaEncodingAttr &mmaLayout,
                                       RankedTensorType type) {
  auto shape = type.getShape();
  auto _warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto rank = shape.size();
  assert(rank == 2 || rank == 3);
  auto warpOrder = triton::gpu::getWarpOrder(mmaLayout);
  ArrayRef<unsigned int> instrShape = mmaLayout.getInstrShape();
  SmallVector<Value> warpsPerCTA;
  for (unsigned i = 0; i < rank; ++i)
    warpsPerCTA.push_back(i32_val(_warpsPerCTA[i]));
  auto shapePerCTA = getShapePerCTA(mmaLayout, shape);

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(32);
  Value laneId = urem(threadId, warpSize);
  Value warpId = udiv(threadId, warpSize);

  uint32_t repM =
      (_warpsPerCTA[rank - 2] * instrShape[rank - 2]) / shapePerCTA[rank - 2];
  uint32_t repN =
      (_warpsPerCTA[rank - 1] * instrShape[rank - 1]) / shapePerCTA[rank - 1];

  uint32_t warpsM;
  if (repM > 1)
    warpsM = _warpsPerCTA[rank - 2] / repM;
  else
    warpsM = shape[rank - 2] / instrShape[rank - 2];

  uint32_t warpsN;
  if (repN > 1)
    warpsN = _warpsPerCTA[rank - 1] / repN;
  else
    warpsN = shape[rank - 1] / instrShape[rank - 1];

  SmallVector<Value> multiDimWarpId(rank);
  multiDimWarpId = delinearize(rewriter, loc, warpId, _warpsPerCTA, warpOrder);
  Value warpIdM = urem(multiDimWarpId[rank - 2], i32_val(warpsM));
  Value warpIdN = urem(multiDimWarpId[rank - 1], i32_val(warpsN));

  Value offWarpM = mul(warpIdM, i32_val(instrShape[rank - 2]));
  Value offWarpN = mul(warpIdN, i32_val(instrShape[rank - 1]));

  SmallVector<Value> multiDimBase(rank);
  if (rank == 3)
    multiDimBase[0] = multiDimWarpId[0];

  // warpsM/N may be 0, in which case warpIDM/N is poison (division by 0), which
  // will cause LLVM to eliminate all ops that depend on the poison value.  This
  // *can* be okay, if the bad dimension is filtered out by a slice layout.  So
  // we rely on the caller to check.  Worst case we crash, which is better than
  // silently producing bad code.
  if (warpsM != 0)
    multiDimBase[rank - 2] = add(udiv(laneId, i32_val(4)), offWarpM);
  if (warpsN != 0)
    multiDimBase[rank - 1] =
        add(mul(i32_val(2), urem(laneId, i32_val(4))), offWarpN);

  return multiDimBase;
}

#ifdef USE_MACA
inline SmallVector<unsigned>
emitSubOffsetForBlockedAndDotLayout(const Attribute &DistributedLayout,
                                    RankedTensorType type,
                                    ArrayRef<int64_t> ctaIdx,
                                    ArrayRef<int64_t> elemIdx) {
  assert(isa<BlockedEncodingAttr>(DistributedLayout) || isa<DotOperandEncodingAttr>(DistributedLayout));
  auto sizePerThread = mlir::triton::gpu::getSizePerThread(DistributedLayout);
  unsigned totalSizePerThreadPerCTA = 1;
  for (int i=0; i < sizePerThread.size(); i++) {
    totalSizePerThreadPerCTA *= sizePerThread[i];
  }
  SmallVector<unsigned> linearIdx;
  for (int i=0; i < ctaIdx.size(); i++) {
    for (int j=0; j < elemIdx.size(); j++) {
      linearIdx.push_back(ctaIdx[i] * totalSizePerThreadPerCTA + elemIdx[j]);
    }
  }
  // sort sudIdx to match subElems for InsertTensor
  std::sort(linearIdx.begin(), linearIdx.end());
  return linearIdx;
}


inline SmallVector<unsigned>
emitSubOffsetForMmaLayoutMACA(const MACAMmaEncodingAttr &mmaLayout,
                              RankedTensorType type,
                              ArrayRef<int64_t> ctaIdx,
                              ArrayRef<int64_t> elemIdx) {
  
  auto shapePerCTA = getShapePerCTATile(mmaLayout);  // elemM*16*warp, elemN*16*warp
  auto threadsPerWarp = getThreadsPerWarp(mmaLayout);  // 4, 16
  auto warpsPerCTA = getWarpsPerCTA(mmaLayout);
  auto elemsM = mmaLayout.getElementsMNK()[0];
  auto elemsN = mmaLayout.getElementsMNK()[1];
  SmallVector<unsigned> elemMN = {elemsM, elemsN};
  unsigned RepM = type.getShape()[0] / shapePerCTA[0];
  unsigned RepN = type.getShape()[1] / shapePerCTA[1];
  SmallVector<unsigned> numReplica = {RepM, RepN};

  SmallVector<unsigned> order;
  if (mmaLayout.getColMajor()) {
    order = {0, 1};
  } else {
    order = {1, 0};
  }
  SmallVector<unsigned> repMVec, repNVec, elemMVec, elemNVec;

  for (int i=0; i<ctaIdx.size(); i++) {
    SmallVector<unsigned> multiCtaId =
        getMultiDimIndex<unsigned>(ctaIdx[i], numReplica, order);
    repMVec.push_back(multiCtaId[0]);
    repNVec.push_back(multiCtaId[1]);
  }
  for (int i=0; i<elemIdx.size(); i++) {
    SmallVector<unsigned> multiElemId =
        getMultiDimIndex<unsigned>(elemIdx[i], elemMN, order);
    elemMVec.push_back(multiElemId[0]);
    elemNVec.push_back(multiElemId[1]);
  }
  SmallVector<unsigned> linearIdx;

  // rowMajor:
  // {elemsN, elemsM, 4, RepN, RepM}
  // {1, elemsN, elemsN * elemsM, elemsN * elemsM * 4, elemsN * elemsM * 4 * RepN}
  // tn + tm * elemsN + depth * elemsN * elemsM + repn * elemsN * elemsM * 4 + repm * elemsN * elemsM * 4 * RepN

  // colMajor:
  // {elemsN, 4, repN, elemsM, repM}
  // {1, elemsN, 4*elemsN, 4*elemsN*repN, 4*elemsN*repN*elemsM}
  unsigned linear;
  for (int i=0; i<repMVec.size(); i++) {
    for (int j=0; j<elemMVec.size(); j++) {
      for (int d=0; d<4; d++) {
        if (!mmaLayout.getColMajor()) {
          linear = elemNVec[j] + elemMVec[j] * elemsN + d * elemsN * elemsM +
                   repNVec[i] * elemsN * elemsM * 4 + repMVec[i] * elemsN * elemsM * 4 * RepN;
        } else {
          linear = elemNVec[j] + d * elemsN + repNVec[i] * 4 * elemsN +
                   elemMVec[j] * 4 * elemsN * RepN + repMVec[i] * 4 * elemsN * RepN * elemsM;
        }
        linearIdx.push_back(linear);
      }
    }
  }
  // sort sudIdx to match subElems for InsertTensor
  std::sort(linearIdx.begin(), linearIdx.end());
  return linearIdx;
}

inline SmallVector<unsigned>
emitSubOffsetForLayout(Attribute layout, RankedTensorType type,
                       ArrayRef<int64_t> ctaIdx,
                       ArrayRef<int64_t> elemIdx) {
  // TODO: check offset for MMA colmajor Layout & DOT Layout 
  if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout))
    return emitSubOffsetForBlockedAndDotLayout(blockedLayout, type, ctaIdx, elemIdx);
  if (auto mmaLayout = dyn_cast<MACAMmaEncodingAttr>(layout))
    return emitSubOffsetForMmaLayoutMACA(mmaLayout, type, ctaIdx, elemIdx);
  if (auto dotLayout = dyn_cast<DotOperandEncodingAttr>(layout))
    return emitSubOffsetForBlockedAndDotLayout(dotLayout, type, ctaIdx, elemIdx);

  llvm_unreachable("unsupported emitOffsetForLayout");
}

inline SmallVector<Value>
emitBaseIndexWithinCTAForDotOperandLayoutMACA(Location loc, RewriterBase &rewriter,
                                       const DotOperandEncodingAttr &dotLayout,
                                       RankedTensorType type) {
  auto shape = type.getShape();
  auto opIdx = dotLayout.getOpIdx();
  auto mmaLayout = 
      dyn_cast<MACAMmaEncodingAttr>(dotLayout.getParent());
  assert(mmaLayout);
  auto _warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto tMN = mmaLayout.getElementsMNK()[opIdx];
  auto tK = mmaLayout.getElementsMNK()[2];
  auto isATrans = mmaLayout.getIsATrans();
  auto isBTrans = mmaLayout.getIsBTrans();
  // MACA: The API is already supported, but it has not been tested yet. 
  // We will verify its correctness when we encounter it in future tests.
  assert((!isATrans) && (!isBTrans));
  auto elementsStride = mmaLayout.getElementsStride();
  auto colMajor = mmaLayout.getColMajor();
  assert(colMajor == 0);
  assert(_warpsPerCTA.size() == 2);
  int warpPerCTA = _warpsPerCTA[opIdx];
  Value warpPerCTA_ = i32_val(warpPerCTA);
  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(64);
  int numThreadMN = 16;
  int numThreadK = 4;
  Value laneId = urem(threadId, warpSize);
  Value warpId = udiv(threadId, warpSize);
  Value remWarpId = urem(urem(warpId, warpPerCTA_), i32_val(shape[opIdx]/(numThreadMN * tMN)));;
  Value offWarpMN = mul(remWarpId, i32_val(numThreadMN * tMN));
  SmallVector<Value> multiDimBase(2);
  // b
  if (opIdx) {
    multiDimBase[0] = mul(i32_val(tK), udiv(laneId, i32_val(numThreadMN)));
    if (isBTrans) {
      auto elemOff = urem(laneId, i32_val(elementsStride[1]));
      auto elemStrideOff = mul(udiv(urem(laneId, i32_val(numThreadMN)), i32_val(elementsStride[1])), 
                          mul(i32_val(tMN), i32_val(elementsStride[1])));
      auto offset = add(elemOff, elemStrideOff);
      multiDimBase[1] = add(add(urem(laneId, i32_val(numThreadMN)), offset), offWarpMN);
    } else {
      multiDimBase[1] = add(mul(urem(laneId, i32_val(numThreadMN)), i32_val(tMN)), offWarpMN);
    }
  // a
  } else {
    if (isATrans) {
      auto elemOff = urem(laneId, i32_val(elementsStride[0]));
      auto elemStrideOff = mul(udiv(urem(laneId, i32_val(numThreadMN)), i32_val(elementsStride[0])), 
                          mul(i32_val(tMN), i32_val(elementsStride[0])));
      auto offset = add(elemOff, elemStrideOff);
      multiDimBase[0] = add(add(offset, urem(laneId, i32_val(numThreadMN))), offWarpMN);
    } else {
      multiDimBase[0] = add(mul(i32_val(tMN), urem(laneId, i32_val(numThreadMN))), offWarpMN);
    }
    multiDimBase[1] = mul(udiv(laneId, i32_val(numThreadMN)), i32_val(tK));
  }
  return multiDimBase;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForDotOperandLayoutMACA(const DotOperandEncodingAttr &dotLayout,
                                  RankedTensorType type) {
  SmallVector<SmallVector<unsigned>> ret;
  auto shape = type.getShape();
  auto opIdx = dotLayout.getOpIdx();
  auto mmaLayout = 
      dyn_cast<MACAMmaEncodingAttr>(dotLayout.getParent());
  auto isATrans = mmaLayout.getIsATrans();
  auto isBTrans = mmaLayout.getIsBTrans();
  auto colMajor = mmaLayout.getColMajor();
  assert(colMajor == 0);
  // MACA: The API is already supported, but it has not been tested yet. 
  // We will verify its correctness when we encounter it in future tests.
  assert((!isATrans) && (!isBTrans));
  auto elementsStride = mmaLayout.getElementsStride();
  assert(mmaLayout);
  auto tMN = mmaLayout.getElementsMNK()[opIdx];
  auto tK = mmaLayout.getElementsMNK()[2];
  int kDim = opIdx ? 0 : 1;
  for (unsigned i = 0; i < shape[opIdx]; i += getShapePerCTATile(dotLayout)[opIdx]) {
    for (unsigned j = 0; j < shape[kDim]; j += getShapePerCTATile(dotLayout)[kDim]) {
      for (unsigned tmn = 0; tmn < tMN; ++tmn) {
        for (unsigned tk = 0; tk < tK; ++tk) {
          if (opIdx) { // b
            ret.push_back({j + tk, i + tmn * elementsStride[1]});
          } else { // a
            ret.push_back({i + tmn * elementsStride[0], j + tk});
          }
        }
      }
    }
  }
  return ret;
}

inline SmallVector<Value>
emitBaseIndexWithinCTAForMmaLayoutMACA(Location loc, RewriterBase &rewriter,
                                       const MACAMmaEncodingAttr &mmaLayout,
                                       RankedTensorType type) {
    auto shape = type.getShape();
    auto _warpsPerCTA = mmaLayout.getWarpsPerCTA();
    auto tM = mmaLayout.getElementsMNK()[0];
    auto tN = mmaLayout.getElementsMNK()[1];
    auto colMajor = mmaLayout.getColMajor();
    auto isATrans = mmaLayout.getIsATrans();
    auto isBTrans = mmaLayout.getIsBTrans();
    // MACA: The API is already supported, but it has not been tested yet. 
    // We will verify its correctness when we encounter it in future tests.
    assert((!isATrans) && (!isBTrans));
    auto elementsStride = mmaLayout.getElementsStride();
    assert(_warpsPerCTA.size() == 2);
    SmallVector<Value> warpsPerCTA = {i32_val(_warpsPerCTA[0]),
                                      i32_val(_warpsPerCTA[1])};
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(64);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    Value warpId0 = urem(urem(warpId, warpsPerCTA[0]), i32_val(shape[0]/(16*tM)));
    Value warpId1 = urem(urem(udiv(warpId, warpsPerCTA[0]), warpsPerCTA[1]),i32_val(shape[1]/(16*tN)));
    Value offWarp0 = mul(warpId0, i32_val(16*tM));
    Value offWarp1 = mul(warpId1, i32_val(16*tN));

    SmallVector<Value> multiDimBase(2);
    if (colMajor != 0) {
      if (isATrans) {
        auto elemOff = urem(laneId, i32_val(elementsStride[0]));
        auto elemStrideOff = mul(udiv(urem(laneId, i32_val(16)), i32_val(elementsStride[0])), 
                            mul(i32_val(tM), i32_val(elementsStride[0])));
        auto offset = add(elemOff, elemStrideOff);
        multiDimBase[0] = add(add(offset, urem(laneId, i32_val(16))), offWarp0);
      } else {
        multiDimBase[0] = add(mul(i32_val(tM), urem(laneId, i32_val(16))), offWarp0);
      }
      multiDimBase[1] = add(mul(udiv(laneId, i32_val(16)), i32_val(4*tN)), offWarp1);
    } else {
      multiDimBase[0] = add(mul(i32_val(4*tM), udiv(laneId, i32_val(16))), offWarp0);
      if (isBTrans) {
        auto elemOff = urem(laneId, i32_val(elementsStride[1]));
        auto elemStrideOff = mul(udiv(urem(laneId, i32_val(16)), i32_val(elementsStride[1])), 
                            mul(i32_val(tN), i32_val(elementsStride[1])));
        auto offset = add(elemOff, elemStrideOff);
        multiDimBase[1] = add(add(urem(laneId, i32_val(16)), offset), offWarp1);
      } else {
        multiDimBase[1] = add(mul(urem(laneId, i32_val(16)), i32_val(tN)), offWarp1);
      }
    }
    return multiDimBase;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForMmaLayoutMACA(const MACAMmaEncodingAttr &mmaLayout,
                            RankedTensorType type) {
  auto shape = type.getShape();
  auto _warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto tM = mmaLayout.getElementsMNK()[0];
  auto tN = mmaLayout.getElementsMNK()[1];
  auto colMajor = mmaLayout.getColMajor();
  auto isATrans = mmaLayout.getIsATrans();
  auto isBTrans = mmaLayout.getIsBTrans();
  auto elementsStride = mmaLayout.getElementsStride();
  SmallVector<SmallVector<unsigned>> ret;
  int rowDepth = tM;
  int colDepth = tN;
  if (colMajor != 0) { 
    colDepth *= 4;
  } else {
    rowDepth *= 4;
  }

  for (unsigned i = 0; i < shape[0]; i += getShapePerCTATile(mmaLayout)[0]) {
    for (unsigned j = 0; j < shape[1]; j += getShapePerCTATile(mmaLayout)[1]) {
      for (unsigned tm = 0; tm < rowDepth; ++tm) {
        for (unsigned tn = 0; tn < colDepth; ++tn) {
          if (colMajor != 0) {
            ret.push_back({i + tm * elementsStride[0], j + tn});
          } else {
            ret.push_back({i + tm, j + tn * elementsStride[1]});
          }
        }
      }
    }
  }
  return ret;
}

#endif

inline SmallVector<SmallVector<unsigned>>
emitOffsetForMmaLayoutV3(const NvidiaMmaEncodingAttr &mmaLayout,
                         RankedTensorType type) {
  auto shape = type.getShape();
  auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
  SmallVector<SmallVector<unsigned>> ret;
  ArrayRef<unsigned int> instrShape = mmaLayout.getInstrShape();

  for (unsigned i = 0; i < shapePerCTA[0];
       i += getShapePerCTATile(mmaLayout)[0]) {
    for (unsigned j = 0; j < shapePerCTA[1];
         j += getShapePerCTATile(mmaLayout)[1]) {
      for (unsigned k = 0; k < instrShape[1]; k += 8) {
        ret.push_back({i, j + k});
        ret.push_back({i, j + k + 1});
        ret.push_back({i + 8, j + k});
        ret.push_back({i + 8, j + k + 1});
      }
    }
  }
  return ret;
}

inline SmallVector<Value>
emitBaseIndexForMfmaLayout(Location loc, RewriterBase &rewriter,
                           const AMDMfmaEncodingAttr &mfmaLayout,
                           RankedTensorType type) {
  auto shape = type.getShape();
  auto rank = shape.size();
  assert(rank == 2 || rank == 3);
  auto _warpsPerCTA = mfmaLayout.getWarpsPerCTA();
  SmallVector<Value> warpsPerCTA;
  for (unsigned i = 0; i < rank; ++i)
    warpsPerCTA.push_back(i32_val(_warpsPerCTA[i]));
  unsigned mDim = mfmaLayout.getMDim();
  unsigned nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(mfmaLayout));
  Value effectiveWarpSize = warpSize;
  if (mDim == 4 && nDim == 4) {
    const int uniqueValuesPerWarp = 4;
    effectiveWarpSize = i32_val(uniqueValuesPerWarp);
  }
  Value laneId = urem(threadId, effectiveWarpSize);
  Value warpId = udiv(threadId, warpSize);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, _warpsPerCTA,
                  triton::gpu::getWarpOrder(mfmaLayout));
  if (shape[rank - 2] >= mDim) {
    assert(shape[rank - 2] % mDim == 0);
    multiDimWarpId[rank - 2] =
        urem(multiDimWarpId[rank - 2],
             i32_val(ceil<unsigned>(shape[rank - 2], mDim)));
  }
  if (shape[rank - 1] >= nDim) {
    assert(shape[rank - 1] % nDim == 0);
    multiDimWarpId[rank - 1] =
        urem(multiDimWarpId[rank - 1],
             i32_val(ceil<unsigned>(shape[rank - 1], nDim)));
  }
  Value offWarp0 = mul(multiDimWarpId[rank - 2], i32_val(mDim));
  Value offWarp1 = mul(multiDimWarpId[rank - 1], i32_val(nDim));

  SmallVector<Value> multiDimBase(rank);
  if (mfmaLayout.getIsTransposed()) {
    multiDimBase[rank - 1] =
        add(mul(i32_val(4), udiv(laneId, i32_val(mDim))), offWarp1);
    multiDimBase[rank - 2] = add(urem(laneId, i32_val(mDim)), offWarp0);
  } else {
    multiDimBase[rank - 2] =
        add(mul(i32_val(4), udiv(laneId, i32_val(nDim))), offWarp0);
    multiDimBase[rank - 1] = add(urem(laneId, i32_val(nDim)), offWarp1);
  }
  // TODO(Lixun): It is assumed when rank = 3, warpsPerCTA is set to
  // {numWarps, 1, 1}. We need to generalize the offset computation.
  if (rank == 3) {
    assert(_warpsPerCTA[1] == 1 && _warpsPerCTA[2] == 1);
    multiDimBase[0] = urem(warpId, i32_val(shape[0]));
  }
  return multiDimBase;
}

inline void emitMfmaOffsetForCTA(const AMDMfmaEncodingAttr &mfmaLayout,
                                 SmallVector<SmallVector<unsigned>> &offsets,
                                 unsigned bOff, unsigned ctaOffsetX,
                                 unsigned ctaOffsetY) {
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  // MFMA output tile consists of repeated "dot operand B" layout groups along
  // row axis. This variable defines number of these groups.
  DenseMap<int, int> groups{{4, 1}, {16, 1}, {32, 4}};
  unsigned numGroups = groups.at(std::min(mDim, nDim));
  const unsigned elemsPerThreadPerGroup = 4;
  auto warpSize = getWarpSize(mfmaLayout);
  assert(warpSize == 64);
  auto shapePerCta = getShapePerCTATile(mfmaLayout);
  auto rank = shapePerCta.size();
  SmallVector<unsigned> elemOff(rank, 0);
  for (unsigned block = 0; block < numGroups; block++) {
    unsigned rowOrColOffset =
        block * elemsPerThreadPerGroup * warpSize / std::min(mDim, nDim);
    for (unsigned elem = 0; elem < elemsPerThreadPerGroup; elem++) {
      if (mfmaLayout.getIsTransposed()) {
        elemOff[rank - 2] = ctaOffsetX * shapePerCta[rank - 2];
        elemOff[rank - 1] =
            ctaOffsetY * shapePerCta[rank - 1] + elem + rowOrColOffset;
      } else {
        elemOff[rank - 2] =
            ctaOffsetX * shapePerCta[rank - 2] + elem + rowOrColOffset;
        elemOff[rank - 1] = ctaOffsetY * shapePerCta[rank - 1];
      }
      if (rank == 3)
        elemOff[0] = bOff;
      offsets.push_back(elemOff);
    }
  }
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForMfmaLayout(const AMDMfmaEncodingAttr &mfmaLayout,
                        RankedTensorType type) {
  auto tensorShape = type.getShape();
  SmallVector<SmallVector<unsigned>> offsets;
  auto shapePerCTA = getShapePerCTA(mfmaLayout, tensorShape);
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
  auto rank = type.getRank();
  SmallVector<unsigned> numReps(rank);
  unsigned mDim = mfmaLayout.getMDim();
  unsigned nDim = mfmaLayout.getNDim();
  assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  SmallVector<unsigned> shapePerWarp(rank, 1);
  shapePerWarp[rank - 2] = mDim;
  shapePerWarp[rank - 1] = nDim;
  for (unsigned d = 0; d < rank; ++d) {
    unsigned inPerCTA = std::min<unsigned>(tensorShape[d], shapePerCTA[d]);
    unsigned inPerWarp = ceil<unsigned>(inPerCTA, warpsPerCTA[d]);
    numReps[d] = ceil<unsigned>(inPerWarp, shapePerWarp[d]);
  }

  unsigned repBatch = rank == 3 ? numReps[0] : 1;
  auto warpsPerBatch =
      rank == 3 ? std::min<unsigned>(tensorShape[0], warpsPerCTA[0]) : 1;

  for (unsigned b = 0; b < repBatch; ++b) {
    for (unsigned i = 0; i < numReps[rank - 2]; ++i) {
      for (unsigned j = 0; j < numReps[rank - 1]; ++j) {
        emitMfmaOffsetForCTA(mfmaLayout, offsets, b * warpsPerBatch, i, j);
      }
    }
  }
  return offsets;
}

inline void emitWmmaOffsetForCTA(const AMDWmmaEncodingAttr &wmmaLayout,
                                 SmallVector<SmallVector<unsigned>> &offsets,
                                 unsigned ctaBatchOffset, unsigned ctaOffsetX,
                                 unsigned ctaOffsetY) {
  const unsigned elemsPerThreadPerGroup = 8;
  auto warpSize = getWarpSize(wmmaLayout);
  assert(warpSize == 32);
  auto shapePerCta = getShapePerCTATile(wmmaLayout);
  auto rank = shapePerCta.size();
  assert(rank == 2 || rank == 3);
  SmallVector<unsigned> elemOffset(rank, 0);
  if (rank == 3)
    elemOffset[0] = ctaBatchOffset;
  for (unsigned elem = 0; elem < elemsPerThreadPerGroup; elem++) {
    elemOffset[rank - 2] = ctaOffsetX * shapePerCta[rank - 2] + 2 * elem;
    elemOffset[rank - 1] = ctaOffsetY * shapePerCta[rank - 1];
    offsets.push_back(elemOffset);
  }
}

inline SmallVector<Value>
emitBaseIndexForWmmaLayout(Location loc, RewriterBase &rewriter,
                           const AMDWmmaEncodingAttr &wmmaLayout,
                           RankedTensorType type) {
  auto shape = type.getShape();
  auto _warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto rank = _warpsPerCTA.size();
  assert(rank == 2 || rank == 3);
  SmallVector<Value> warpsPerCTA;
  for (unsigned i = 0; i < rank; ++i)
    warpsPerCTA.push_back(i32_val(_warpsPerCTA[i]));
  auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr();

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(wmmaLayout));
  Value laneId =
      urem(threadId, i32_val(triton::gpu::getWarpSize(wmmaLayout) / 2));
  Value threadIdPerWarp = urem(threadId, warpSize);

  Value warpId = udiv(threadId, warpSize);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, _warpsPerCTA,
                  triton::gpu::getWarpOrder(wmmaLayout));
  if (shape[rank - 2] >= mnkDim[0]) {
    assert(shape[rank - 2] % mnkDim[0] == 0);
    multiDimWarpId[rank - 2] =
        urem(multiDimWarpId[rank - 2],
             i32_val(ceil<unsigned>(shape[rank - 2], mnkDim[0])));
  }
  if (shape[rank - 1] >= mnkDim[1]) {
    assert(shape[rank - 1] % mnkDim[1] == 0);
    multiDimWarpId[rank - 1] =
        urem(multiDimWarpId[rank - 1],
             i32_val(ceil<unsigned>(shape[rank - 1], mnkDim[1])));
  }
  Value offWarp0 = mul(multiDimWarpId[rank - 2], i32_val(mnkDim[0]));
  Value offWarp1 = mul(multiDimWarpId[rank - 1], i32_val(mnkDim[1]));

  SmallVector<Value> multiDimBase(rank);

  multiDimBase[rank - 2] =
      add(udiv(threadIdPerWarp, i32_val(mnkDim[2])), offWarp0);
  multiDimBase[rank - 1] = add(laneId, offWarp1);

  // TODO: It is assumed when rank = 3, warpsPerCTA is set to
  // {numWarps, 1, 1}. We need to generalize the offset computation.
  if (rank == 3) {
    assert(_warpsPerCTA[1] == 1 && _warpsPerCTA[2] == 1);
    multiDimBase[0] = urem(warpId, i32_val(shape[0]));
  }
  return multiDimBase;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForWmmaLayout(const AMDWmmaEncodingAttr &wmmaLayout,
                        RankedTensorType type) {
  auto tensorShape = type.getShape();
  SmallVector<SmallVector<unsigned>> offsets;
  auto shapePerCTA = getShapePerCTA(wmmaLayout, tensorShape);
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();

  auto rank = tensorShape.size();
  assert(rank == 2 || rank == 3);

  SmallVector<unsigned> numWarpsPerDim(rank, 1);
  auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr();
  SmallVector<unsigned> shapePerWarp(rank, 1);
  shapePerWarp[rank - 2] = mnkDim[0];
  shapePerWarp[rank - 1] = mnkDim[1];
  for (unsigned d = 0; d < rank; ++d) {
    unsigned inPerCTA = std::min<unsigned>(tensorShape[d], shapePerCTA[d]);
    unsigned inPerWarp = ceil<unsigned>(inPerCTA, warpsPerCTA[d]);
    numWarpsPerDim[d] = ceil<unsigned>(inPerWarp, shapePerWarp[d]);
  }

  unsigned repBatch = rank == 3 ? numWarpsPerDim[0] : 1;
  unsigned repM = numWarpsPerDim[rank - 2];
  unsigned repN = numWarpsPerDim[rank - 1];
  auto warpsPerBatch =
      rank == 3 ? std::min<unsigned>(tensorShape[0], warpsPerCTA[0]) : 1;

  for (unsigned b = 0; b < repBatch; ++b) {
    for (unsigned i = 0; i < repM; ++i) {
      for (unsigned j = 0; j < repN; ++j) {
        emitWmmaOffsetForCTA(wmmaLayout, offsets, b * warpsPerBatch, i, j);
      }
    }
  }
  return offsets;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type);

inline SmallVector<SmallVector<unsigned>>
emitOffsetForSliceLayout(const SliceEncodingAttr &sliceLayout,
                         RankedTensorType type) {
  auto parentEncoding = sliceLayout.getParent();
  unsigned dim = sliceLayout.getDim();
  auto parentShape = sliceLayout.paddedShape(type.getShape());
  RankedTensorType parentTy =
      RankedTensorType::get(parentShape, type.getElementType(), parentEncoding);
  auto parentOffsets = emitOffsetForLayout(parentEncoding, parentTy);
  if (parentOffsets.empty())
    return {};

  SmallVector<SmallVector<unsigned>> resultOffsets;
  std::set<SmallVector<unsigned>> uniqueOffsets;

  for (unsigned i = 0; i < parentOffsets.size(); ++i) {
    SmallVector<unsigned> offsets(parentOffsets[i].begin(),
                                  parentOffsets[i].end());
    offsets.erase(offsets.begin() + dim);
    if (auto [it, inserted] = uniqueOffsets.insert(offsets); inserted) {
      resultOffsets.push_back(offsets);
    }
  }

  // It can happen that after deduplicating elements above, resultOffsets has
  // fewer than getTotalElementsPerThread() elements.  In that case repeat the
  // sequence.
  int elemsPerThread = triton::gpu::getTotalElemsPerThread(type);
  assert(resultOffsets.size() > 0);
  assert(elemsPerThread % resultOffsets.size() == 0);
  int numRepeats = elemsPerThread / resultOffsets.size();
  SmallVector<SmallVector<unsigned>> ret;
  for (int i = 0; i < numRepeats; ++i) {
    for (unsigned j = 0; j < resultOffsets.size(); ++j) {
      ret.push_back(SmallVector<unsigned>(resultOffsets[j]));
    }
  }
  return ret;
}

// -----------------------------------------------------------------------
// Get offsets / indices for any layout
// -----------------------------------------------------------------------

inline SmallVector<Value> emitCTAOffsetForLayout(Location loc,
                                                 RewriterBase &rewriter,
                                                 const TargetInfoBase &target,
                                                 Attribute layout,
                                                 ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  SmallVector<unsigned> CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
  SmallVector<unsigned> CTASplitNum = triton::gpu::getCTASplitNum(layout);
  SmallVector<unsigned> CTAOrder = triton::gpu::getCTAOrder(layout);
  SmallVector<int64_t> shapePerCTA =
      triton::gpu::getShapePerCTA(CTASplitNum, shape);

  // Delinearize clusterCTAId
  Value clusterCTAId = target.getClusterCTAId(rewriter, loc);
  SmallVector<Value> multiDimClusterCTAId =
      delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

  // CTA Wrapping
  for (unsigned i = 0; i < rank; ++i) {
    // This wrapping rule must be consistent with getShapePerCTA
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    multiDimClusterCTAId[i] = urem(multiDimClusterCTAId[i], i32_val(splitNum));
  }

  SmallVector<Value> CTAOffset(rank);
  for (unsigned i = 0; i < rank; ++i)
    CTAOffset[i] = mul(multiDimClusterCTAId[i], i32_val(shapePerCTA[i]));

  return CTAOffset;
}

inline SmallVector<Value>
emitBaseIndexForLayoutImpl(Location loc, RewriterBase &rewriter,
                           const TargetInfoBase &target, Attribute layout,
                           RankedTensorType type, bool withCTAOffset) {
  auto shape = type.getShape();

  SmallVector<Value> baseIndex;
  RewriterBase::InsertionGuard guard(rewriter);
  SmallVector<Value> result;
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(layout)) {
    result = emitBaseIndexWithinCTAForBlockedLayout(loc, rewriter,
                                                    blockedLayout, type);
  } else if (auto mmaLayout = mlir::dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
    if (mmaLayout.isVolta())
      result =
          emitBaseIndexWithinCTAForMmaLayoutV1(loc, rewriter, mmaLayout, type);
    if (mmaLayout.isAmpere() || mmaLayout.isHopper())
      result = emitBaseIndexWithinCTAForMmaLayoutV2V3(loc, rewriter, mmaLayout,
                                                      type);
#ifdef USE_MACA
  } else if (auto mmaLayout = mlir::dyn_cast<MACAMmaEncodingAttr>(layout)) {
    result = emitBaseIndexWithinCTAForMmaLayoutMACA(loc, rewriter, mmaLayout,
                                                      type);
  } else if (auto dot = mlir::dyn_cast<DotOperandEncodingAttr>(layout)) {
    if (isa<MACAMmaEncodingAttr>(dot.getParent())) {
      result = emitBaseIndexWithinCTAForDotOperandLayoutMACA(loc, rewriter, dot,
                                                             type);
    } else {
      llvm_unreachable("unsupported emitBaseIndexForLayout");
    }
#endif
  } else if (auto mfmaLayout = mlir::dyn_cast<AMDMfmaEncodingAttr>(layout)) {
    result = emitBaseIndexForMfmaLayout(loc, rewriter, mfmaLayout, type);
  } else if (auto wmmaLayout = mlir::dyn_cast<AMDWmmaEncodingAttr>(layout)) {
    result = emitBaseIndexForWmmaLayout(loc, rewriter, wmmaLayout, type);
  } else if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(type.getShape());
    RankedTensorType parentTy =
        RankedTensorType::get(parentShape, type.getElementType(), parentLayout);
    result = emitBaseIndexForLayoutImpl(loc, rewriter, target, parentLayout,
                                        parentTy, withCTAOffset);
    result.erase(result.begin() + sliceLayout.getDim());
    // CTAOffset has been added in emitBaseIndexForLayout of parentLayout
    return result;
  } else {
    llvm_unreachable("unsupported emitBaseIndexForLayout");
  }
  if (withCTAOffset) {
    auto CTAOffset =
        emitCTAOffsetForLayout(loc, rewriter, target, layout, shape);
    assert(CTAOffset.size() == result.size() && "Rank mismatch");
    for (unsigned k = 0; k < result.size(); ++k) {
      // Individual elements of `result` may be null.  In the caller
      // (emitBaseIndexForLayout), we assert that all such dimensions are sliced
      // off.
      if (!result[k])
        continue;
      result[k] = add(result[k], CTAOffset[k]);
    }
  }
  return result;
}

inline SmallVector<Value>
emitBaseIndexForLayout(Location loc, RewriterBase &rewriter,
                       const TargetInfoBase &target, Attribute layout,
                       RankedTensorType type, bool withCTAOffset) {
  SmallVector<Value> idx = emitBaseIndexForLayoutImpl(
      loc, rewriter, target, layout, type, withCTAOffset);

  // Check that any null values were sliced out.
  for (Value v : idx) {
    if (!v) {
      llvm::errs() << "Failed to generate indexing code, possibly due to bad "
                      "#mma layout.  Please rerun your program with "
                      "MLIR_ENABLE_DUMP=1 and file a bug."
                   << "\nloc: " << loc << "\nlayout: " << layout
                   << "\ntype: " << type << "\nwithCTAOffset: " << withCTAOffset
                   << "\n";
      llvm::report_fatal_error("Failed to generate indexing code");
    }
  }

  return idx;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type) {
  if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout))
    return emitOffsetForBlockedLayout(blockedLayout, type);
  if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
    if (mmaLayout.isVolta())
      return emitOffsetForMmaLayoutV1(mmaLayout, type);
    if (mmaLayout.isAmpere())
      return emitOffsetForMmaLayoutV2(mmaLayout, type);
    if (mmaLayout.isHopper())
      return emitOffsetForMmaLayoutV3(mmaLayout, type);
  }
  if (auto mfmaLayout = mlir::dyn_cast<AMDMfmaEncodingAttr>(layout)) {
    return emitOffsetForMfmaLayout(mfmaLayout, type);
  }
  if (auto wmmaLayout = mlir::dyn_cast<AMDWmmaEncodingAttr>(layout)) {
    return emitOffsetForWmmaLayout(wmmaLayout, type);
  }
#ifdef USE_MACA
  if (auto mmaLayout = mlir::dyn_cast<MACAMmaEncodingAttr>(layout)) {
    return emitOffsetForMmaLayoutMACA(mmaLayout, type);
  }
  if (auto dot = mlir::dyn_cast<DotOperandEncodingAttr>(layout)) {
    if (isa<MACAMmaEncodingAttr>(dot.getParent())) {
      return emitOffsetForDotOperandLayoutMACA(dot, type);
    } else {
      llvm_unreachable("unsupported emitBaseIndexForLayout");
    }
  }
#endif
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout))
    return emitOffsetForSliceLayout(sliceLayout, type);
  llvm_unreachable("unsupported emitOffsetForLayout");
}

// Eventually this will become the only emitIndices function.
std::optional<SmallVector<SmallVector<Value>>>
emitIndicesUsingLinearLayouts(Location loc, RewriterBase &rewriter,
                              const TargetInfoBase &target, Attribute layout,
                              RankedTensorType type, bool withCTAOffset);

// Emit indices calculation within each ConversionPattern, and returns a
// [elemsPerThread X rank] index matrix.
inline SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
            Attribute layout, RankedTensorType type, bool withCTAOffset,
            bool allowLL = true) {
  // Eventually the LinearLayout path will be the only one.  For now we allow
  // both paths so we can test that they produce the same results.
  if (allowLL && target.enableLinearLayout()) {
    std::optional<SmallVector<SmallVector<Value>>> llOffsets =
        emitIndicesUsingLinearLayouts(loc, rewriter, target, layout, type,
                                      withCTAOffset);
    if (llOffsets.has_value())
      return *llOffsets;
  }

  // step 1, delinearize threadId to get the base index
  auto multiDimBase = emitBaseIndexForLayout(loc, rewriter, target, layout,
                                             type, withCTAOffset);
  // step 2, get offset of each element
  auto offset = emitOffsetForLayout(layout, type);
  // step 3, add offset to base, and reorder the sequence
  // of indices to guarantee that elems in the same
  // sizePerThread are adjacent in order
  auto shape = type.getShape();
  unsigned rank = shape.size();
  unsigned elemsPerThread = offset.size();
  SmallVector<SmallVector<Value>> multiDimIdx(elemsPerThread,
                                              SmallVector<Value>(rank));
  for (unsigned n = 0; n < elemsPerThread; ++n)
    for (unsigned k = 0; k < rank; ++k)
      multiDimIdx[n][k] = add(multiDimBase[k], i32_val(offset[n][k]));

  return multiDimIdx;
}

/* ---------------- */
/* ---------------- */
inline DenseMap<unsigned, Value> getSwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::SharedEncodingAttr resSharedLayout,
    Type resElemTy, SharedMemoryObject smemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals, SmallVectorImpl<Value> &srcStrides) {
  // This utility computes the pointers for accessing the provided swizzled
  // shared memory layout `resSharedLayout`. More specifically, it computes,
  // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
  // the pointer: ptr[(row, col)] = base + (rowOff * strides[ord[1]] +
  // colOff) where :
  //   phase = (row // perPhase) % maxPhase
  //   rowOff = row
  //   colOff = colOffSwizzled + colOffOrdered
  //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
  //     colOffOrdered = (col % outVec) // minVec * minVec
  //
  // Note 1:
  // -------
  // Because swizzling happens at a granularity of outVec, we need to
  // decompose the offset into a swizzled factor and a non-swizzled
  // (ordered) factor
  //
  // Note 2:
  // -------
  // If we have x, y, z of the form:
  // x = 0b00000xxxx
  // y = 0byyyyy0000
  // z = 0b00000zzzz
  // then (x + y) XOR z = 0byyyyxxxx XOR 0b00000zzzz = (x XOR z) + y
  // This means that we can use some immediate offsets for shared memory
  // operations.
#ifndef USE_MACA_OPAQUE_PTR
  auto dstPtrTy = ptr_ty(rewriter.getContext(), resElemTy, 3);
#else
  auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
  auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
#ifndef USE_MACA_OPAQUE_PTR
  Value dstPtrBase = bitcast(gep(dstPtrTy, resElemTy, smemObj.base, dstOffset), dstPtrTy);
#else
  Value dstPtrBase = gep(dstPtrTy, resElemTy, smemObj.base, dstOffset);
#endif

  auto srcEncoding = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  // swizzling params as described in TritonGPUAttrDefs.td
  unsigned outVec = resSharedLayout.getVec();
  unsigned perPhase = resSharedLayout.getPerPhase();
  unsigned maxPhase = resSharedLayout.getMaxPhase();
  // Order
  auto inOrder = triton::gpu::getOrder(srcEncoding);
  auto outOrder = triton::gpu::getOrder(resSharedLayout);
  assert(maxPhase == 1 ||
         outVec * maxPhase <= srcShape[outOrder[0]] &&
             "Swizzling would generate out of bounds memory accesses");
  // Tensor indices held by the current thread, as LLVM values
  auto srcIndices = emitIndices(loc, rewriter, target, srcEncoding, srcTy,
                                /*withCTAOffset=*/false);
  // Swizzling with leading offsets (e.g. Hopper GMMA)
  unsigned swizzlingByteWidth = 0;
  if (resSharedLayout.getHasLeadingOffset()) {
    if (perPhase == 4 && maxPhase == 2)
      swizzlingByteWidth = 32;
    else if (perPhase == 2 && maxPhase == 4)
      swizzlingByteWidth = 64;
    else if (perPhase == 1 && maxPhase == 8)
      swizzlingByteWidth = 128;
    else
      llvm::report_fatal_error("Unsupported shared layout.");
  }
  unsigned numElemsPerSwizzlingRow =
      swizzlingByteWidth * 8 / resElemTy.getIntOrFloatBitWidth();
  Value numElemsPerSwizzlingRowVal = i32_val(numElemsPerSwizzlingRow);
  unsigned leadingDimOffset;
  if (outOrder.size() >= 2) {
    leadingDimOffset = numElemsPerSwizzlingRow * srcShapePerCTA[outOrder[1]];
  } else {
    leadingDimOffset = numElemsPerSwizzlingRow;
  }

  Value leadingDimOffsetVal = i32_val(leadingDimOffset);
  // Return values
  DenseMap<unsigned, Value> ret;
  // cache for non-immediate offsets
  DenseMap<unsigned, Value> cacheCol, cacheRow;
  unsigned minVec = std::min(outVec, inVec);
  Value strideRow = outOrder.size() >= 2 ? srcStrides[outOrder[1]] : i32_val(0);
  Value strideCol = srcStrides[outOrder[0]];
  LDBG("getSwizzledSharedPtrs: perPhase = "
       << perPhase << " maxPhase = " << maxPhase << " minVec = " << minVec
       << " inVec = " << inVec << " outVec = " << outVec << " strideRow "
       << strideRow << " strideCol " << strideCol);
  for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
    Value offset = i32_val(0);
    // Extract multi dimensional index for current element
    auto idx = srcIndices[elemIdx];
    Value idxCol = idx[outOrder[0]]; // contiguous dimension
    Value idxRow;
    if (outOrder.size() >= 2) {
      idxRow = idx[outOrder[1]]; // discontiguous dimension
    } else {
      idxRow = i32_val(0);
    }

#ifdef USE_MACA
      if (std::getenv("TRITON_DISABLE_SWIZZLE") == nullptr && outOrder.size() == 2) {
        Value offset;
        if (maxPhase == 1) {
          offset = add(mul(idxRow, strideRow), mul(idxCol, strideCol));
        } else if (outVec == minVec) {
          Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
          Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
          colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
          Value colOff = colOffSwizzled;
          offset = add(mul(idxRow, strideRow), mul(colOff, strideCol));
        } else {
          Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
          Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
          colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
          Value colOffOrdered = urem(idxCol, i32_val(outVec));
          colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
          colOffOrdered = mul(colOffOrdered, i32_val(minVec));
          Value colOff = add(colOffSwizzled, colOffOrdered);
          offset = add(mul(idxRow, strideRow), mul(colOff, strideCol));
        }
        Value currPtr = bitcast(gep(dstPtrTy, resElemTy, dstPtrBase, offset), dstPtrTy);
        ret[elemIdx] = currPtr;
        continue;
      }
#endif

    // compute phase = (row // perPhase) % maxPhase
    Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
    // extract dynamic/static offset for immediate offsetting
    unsigned immedateOffCol = 0;
    unsigned immedateOffRow = 0;
    if (leadingDimOffset) {
      // hopper
      offset =
          mul(udiv(idxCol, numElemsPerSwizzlingRowVal), leadingDimOffsetVal);
      // Shrink by swizzling blocks
      idxCol = urem(idxCol, numElemsPerSwizzlingRowVal);
      strideRow = numElemsPerSwizzlingRowVal;
    }
    if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxCol.getDefiningOp())) {
      if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
              add.getRhs().getDefiningOp())) {
        unsigned cst =
            cast<IntegerAttr>(_cst.getValue()).getValue().getSExtValue();
        unsigned key = cst % (outVec * maxPhase);
        cacheCol.insert({key, idxCol});
        idxCol = cacheCol[key];
        immedateOffCol = cst / (outVec * maxPhase) * (outVec * maxPhase);
      }
    }
    if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxRow.getDefiningOp())) {
      if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
              add.getRhs().getDefiningOp())) {
        unsigned cst =
            mlir::cast<IntegerAttr>(_cst.getValue()).getValue().getSExtValue();
        unsigned key = cst % (perPhase * maxPhase);
        cacheRow.insert({key, idxRow});
        idxRow = cacheRow[key];
        immedateOffRow = cst / (perPhase * maxPhase) * (perPhase * maxPhase);
      }
    }
    // row offset is simply row index
    Value rowOff = mul(idxRow, strideRow);
    // because swizzling happens at a granularity of outVec, we need to
    // decompose the offset into a swizzled factor and a non-swizzled
    // (ordered) factor: colOffSwizzled = ((col // outVec) ^ phase) * outVec
    // colOffOrdered = (col % outVec) // minVec * minVec
    Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
    colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
    Value colOffOrdered = urem(idxCol, i32_val(outVec));
    colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
    colOffOrdered = mul(colOffOrdered, i32_val(minVec));
    Value colOff = add(colOffSwizzled, colOffOrdered);
    // compute non-immediate offset
    if (outOrder.size() == 3)
      offset = add(offset, mul(idx[outOrder[2]], srcStrides[outOrder[2]]));
    offset = add(offset, add(rowOff, mul(colOff, strideCol)));
#ifndef USE_MACA_OPAQUE_PTR
    Value currPtr = bitcast(gep(dstPtrTy, resElemTy, dstPtrBase, offset), dstPtrTy);
#else
    Value currPtr = gep(dstPtrTy, resElemTy, dstPtrBase, offset);
#endif
    // compute immediate offset
    Value immediateOff;
    if (outOrder.size() >= 2) {
      immediateOff =
          add(mul(i32_val(immedateOffRow), strideRow), i32_val(immedateOffCol));
    } else {
      immediateOff = i32_val(immedateOffCol);
    }
#ifndef USE_MACA_OPAQUE_PTR
    ret[elemIdx] = bitcast(gep(dstPtrTy, resElemTy, currPtr, immediateOff), dstPtrTy);
#else
    ret[elemIdx] = gep(dstPtrTy, resElemTy, currPtr, immediateOff);
#endif
  }
  return ret;
}

#ifdef USE_MACA
inline void getTargetInsertOp(int targetIndex, LLVM::InsertValueOp &insertOp) {
  while (auto InsertOp1 = insertOp.getODSOperands(0)[0].getDefiningOp<LLVM::InsertValueOp>()) {
    auto InsertOp1Pos = InsertOp1.getPosition();
    if (targetIndex == InsertOp1Pos[0]) {
      insertOp = InsertOp1;
      break;
    }
    insertOp = InsertOp1;
  }
}

// recursively backward retrive cmpOp0 
inline DenseMap<unsigned, Value>
parseAndOpLhs(Value andOpLhs) {
  DenseMap<unsigned, Value> maskValueAndOp;
  if (auto extractOp1 = andOpLhs.getDefiningOp<LLVM::ExtractValueOp>()) {
    debug_print << "parseAndOpLhs extractOp1: " << extractOp1 << "\n";
    auto extractOp1Pos = extractOp1.getPosition();
    if (auto InsertOp2 = extractOp1.getODSOperands(0)[0].getDefiningOp<LLVM::InsertValueOp>()) {
      debug_print << "parseAndOpLhs InsertOp2: " << InsertOp2 << "\n";
      getTargetInsertOp(extractOp1Pos[0], InsertOp2);
      if (auto cmpOp0 = InsertOp2.getODSOperands(1)[0].getDefiningOp<LLVM::ICmpOp>()) {
        debug_print << "parseAndOpLhs cmpOp0: " << cmpOp0 << "\n";
        auto cmpOp0Lhs = cmpOp0.getLhs();
        auto cmpOp0Rhs = cmpOp0.getRhs();
        maskValueAndOp[0] = cmpOp0Lhs;
        maskValueAndOp[1] = cmpOp0Rhs;
        return maskValueAndOp;
      } else if (auto extractOp3 = InsertOp2.getODSOperands(1)[0].getDefiningOp<LLVM::ExtractValueOp>()) {
        debug_print << "parseAndOpLhs extractOp3: " << extractOp3 << "\n";
        return parseAndOpLhs(InsertOp2.getODSOperands(1)[0]);
      } else if (auto andOp = InsertOp2.getODSOperands(1)[0].getDefiningOp<LLVM::AndOp>()) {
        debug_print << "parseAndOpLhs andOp: " << andOp << "\n";
        return parseAndOpLhs(andOp.getLhs());
      } else {
        debug_print << "parseAndOpLhs InsertOp2(1)[0] not ICmpOp, InsertOp2(1)[0]: " << InsertOp2.getODSOperands(1)[0] << "\n";
      }
    } else {
      debug_print << "parseAndOpLhs extractOp1 not InsertValueOp, extractOp1: " << extractOp1 << "\n";
    }
  } else {
    debug_print << "parseAndOpLhs andOpLhs not ExtractValueOp, andOpLhs: " << andOpLhs << "\n";
  }
  return maskValueAndOp;
}

inline Value 
getContiguityGvmStartAddr(Location loc, 
                          ConversionPatternRewriter &rewriter,
                          Value gvmPtr,
                          RankedTensorType srcTy,
                          MemDescType resTy,
                          const TypeConverter *converter) {
  auto resSharedLayout = cast<triton::gpu::SharedEncodingAttr>(resTy.getEncoding());
  unsigned maxPhase = resSharedLayout.getMaxPhase();
  if (maxPhase == 1) {
    return gvmPtr;
  }

  auto resElemTy = converter->convertType(resTy.getElementType());
  auto srcEncoding = srcTy.getEncoding();
  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto inOrder = triton::gpu::getOrder(srcEncoding);
  assert(inOrder.size() == 2 && threadsPerWarp.size() == 2 &&
         "Unexpected Order rank of getContiguityGvmStartAddr");
  int getContiguityDimTNum;
  for (int i = 0; i < inOrder.size(); i++) {
    if (inOrder[i] == 0) {
      getContiguityDimTNum = threadsPerWarp[i];
      break;
    }
  }

  int numElemsPerThread = 0;
  int getContiDimSizePerCTA = 0;
  auto srcBlockedLayout = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
  auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
  auto sizePerThread = triton::gpu::getSizePerThread(srcBlockedLayout);

  for (int i = 0; i < inOrder.size(); i++) {
    if (inOrder[i] == 0) {
      numElemsPerThread = sizePerThread[i];
      getContiDimSizePerCTA = srcShapePerCTA[i];
      break;
    }
  }

  int getContiDimWarpNum = getContiDimSizePerCTA / (numElemsPerThread * getContiguityDimTNum);
  getContiDimWarpNum = getContiDimWarpNum == 0 ? 1 : getContiDimWarpNum;

  Value contiDimTNum = i32_val(64 * getContiDimWarpNum);
  Value threadId = getThreadId(rewriter, loc);
  Value lane = urem(threadId, contiDimTNum);
  Value tidMulti = udiv(lane, i32_val(getContiguityDimTNum * getContiDimWarpNum));
  Value thread = mul(tidMulti, i32_val(getContiguityDimTNum * getContiDimWarpNum));
  Value startThread = sub(thread, lane);
#ifdef USE_MACA_OPAQUE_PTR
  auto dstPtrTy = ptr_ty(rewriter.getContext(), 1);
#else
  auto dstPtrTy = ptr_ty(resElemTy, 1);
#endif
  Value srcOffset = mul(startThread, i32_val(numElemsPerThread));
  Value srcPtrSwiBase = gep(dstPtrTy, resElemTy, gvmPtr, srcOffset);
  return srcPtrSwiBase;
}

inline Value 
getCopyAsyncSwizzledGvmPtrs(Location loc, ConversionPatternRewriter &rewriter,
                            const TargetInfoBase &target,
                            unsigned inVec, unsigned elemIdx, RankedTensorType srcTy,
                            triton::gpu::SharedEncodingAttr resSharedLayout,
                            Type srcElemTy, Value gvmPtr) {
  // This utililty computes the pointers for accessing the provided swizzled
  // shared memory layout `resSharedLayout`. More specifically, it computes,
  // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
  // the pointer: ptr[(row, col)] = base + (rowOff * strides[ord[1]] +
  // colOff) where :
  //   compute phase = (row // perPhase) % maxPhase
  //   rowOff = row
  //   colOff = colOffSwizzled + colOffOrdered
  //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
  //     colOffOrdered = (col % outVec) // minVec * minVec
  //
  // Note 1:
  // -------
  // Because swizzling happens at a granularity of outVec, we need to
  // decompose the offset into a swizzled factor and a non-swizzled
  // (ordered) factor
  //
  // Note 2:
  // -------
  // If we have x, y, z of the form:
  // x = 0b00000xxxx
  // y = 0byyyyy0000
  // z = 0b00000zzzz
  // then (x + y) XOR z = 0byyyyxxxx XOR 0b00000zzzz = (x XOR z) + y
  // This means that we can use some immediate offsets for shared memory
  // operations.

  auto srcEncoding = srcTy.getEncoding();
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);

  // swizzling params as described in TritonGPUAttrDefs.td
  unsigned outVec = resSharedLayout.getVec();
  unsigned perPhase = resSharedLayout.getPerPhase();
  unsigned maxPhase = resSharedLayout.getMaxPhase();
  unsigned minVec = std::min(outVec, inVec);

  // order
  auto outOrder = triton::gpu::getOrder(resSharedLayout);

  // tensor indices held by the current thread, as LLVM values
  auto srcIndices = emitIndices(loc, rewriter, target, srcEncoding, srcTy,
                                /*withCTAOffset=*/false);

  // extract multi dimensional index for current element
  assert((elemIdx < numElems) && "elemIdx needs to be less than numElems");
  auto idx = srcIndices[elemIdx];
  Value idxCol = idx[outOrder[0]]; // contiguous dimension
  Value idxRow = idx[outOrder[1]]; // discontiguous dimension
#ifdef USE_MACA_OPAQUE_PTR
  auto srcPtrTy = ptr_ty(rewriter.getContext(), 1);
#else
  auto srcPtrTy = ptr_ty(srcElemTy, 1);
#endif
  
  Value srcSwiPtr;
  if (maxPhase == 1) {
    srcSwiPtr = gvmPtr;
  } else if (outVec == minVec) {
    Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
    Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
    colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
    Value colOff = colOffSwizzled;
    Value currPtr = gep(srcPtrTy, srcElemTy, gvmPtr, colOff);
    srcSwiPtr = currPtr;
  } else {
    Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
    Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
    colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
    Value colOffOrdered = urem(idxCol, i32_val(outVec));
    colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
    colOffOrdered = mul(colOffOrdered, i32_val(minVec));
    Value colOff = add(colOffSwizzled, colOffOrdered);
    Value currPtr = gep(srcPtrTy, srcElemTy, gvmPtr, colOff);
    srcSwiPtr = currPtr;
  }
  return srcSwiPtr;
}

#endif

inline SmallVector<Value> loadSharedToDistributed(
    Value dst, Value src, SharedMemoryObject smemObj, Type elemTy, Location loc,
    ConversionPatternRewriter &rewriter, const TargetInfoBase &target) {
  auto dstTy = cast<RankedTensorType>(dst.getType());
  auto dstShape = dstTy.getShape();
  assert(dstShape.size() <= 2 && "Unexpected rank of loadSharedToDistributed");
  auto srcTy = cast<MemDescType>(src.getType());
  auto dstDistributedLayout = dstTy.getEncoding();
  if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(dstDistributedLayout)) {
    assert((!mmaLayout.isVolta()) &&
           "ConvertLayout Shared->MMAv1 is not supported yet");
  }
  auto srcSharedLayout =
      cast<triton::gpu::SharedEncodingAttr>(srcTy.getEncoding());
  auto srcElemTy = srcTy.getElementType();
  auto dstElemTy = dstTy.getElementType();
  LDBG("loadSharedToDistributed elemTy " << elemTy << " srcElemTy " << srcElemTy
                                         << " dstElemTy " << dstElemTy);
  auto inOrd = triton::gpu::getOrder(srcSharedLayout);
  auto outOrd = triton::gpu::getOrder(dstDistributedLayout);
  unsigned outVec = inOrd == outOrd
                        ? triton::gpu::getUniqueContigPerThread(
                              dstDistributedLayout, dstShape)[outOrd[0]]
                        : 1;

  // If the shmem layout is not swizzled, we can trivially vectorize loads
  // across the whole width of the most-minor dimension of the shape, because
  // Triton requires all the dims are powers of 2.
  unsigned inVec = srcSharedLayout.getMaxPhase() == 1
                       ? srcTy.getShape()[inOrd[0]]
                       : srcSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
  SmallVector<Value> offsetVals = {smemObj.strides.size(), i32_val(0)};

  DenseMap<unsigned, Value> sharedPtrs =
      getSwizzledSharedPtrs(loc, target, outVec, dstTy, srcSharedLayout, elemTy,
                            smemObj, rewriter, offsetVals, smemObj.strides);
  assert(outElems % minVec == 0 && "Unexpected number of elements");
  unsigned numVecs = outElems / minVec;
  auto wordTy = vec_ty(elemTy, minVec);
  SmallVector<Value> outVals(outElems);
  for (unsigned i = 0; i < numVecs; ++i) {
    Value smemAddr = sharedPtrs[i * minVec];
  #ifndef USE_MACA_OPAQUE_PTR
    smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), wordTy, 3));
  #else
    smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3)); 
  #endif
    auto valVec = load(wordTy, smemAddr);
    valVec.setAlignment(minVec * elemTy.getIntOrFloatBitWidth() / 8);
    for (unsigned v = 0; v < minVec; ++v) {
      Value currVal = extract_element(elemTy, valVec, i32_val(v));
      outVals[i * minVec + v] = currVal;
    }
  }
  return outVals;
}

inline void storeDistributedToShared(Value src, ArrayRef<Value> inVals,
                                     ArrayRef<Value> dstStrides, Value dst,
                                     Value smemBase, Type elemTy, Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const TargetInfoBase &target) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto srcShape = srcTy.getShape();
  auto rank = srcShape.size();
  assert(rank <= 3 && "Unexpected rank of storeDistributedToShared");
  auto dstTy = cast<MemDescType>(dst.getType());
  auto srcDistributedLayout = srcTy.getEncoding();
  if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(srcDistributedLayout)) {
    assert((!mmaLayout.isVolta()) &&
           "ConvertLayout MMAv1->Shared is not supported yet");
  }
  auto dstSharedLayout =
      cast<triton::gpu::SharedEncodingAttr>(dstTy.getEncoding());
  auto dstElemTy = dstTy.getElementType();
  auto inOrd = triton::gpu::getOrder(srcDistributedLayout);
  auto outOrd = dstSharedLayout.getOrder();
  unsigned inVec = inOrd == outOrd
                       ? triton::gpu::getUniqueContigPerThread(
                             srcDistributedLayout, srcShape)[inOrd[0]]
                       : 1;
  // If the shmem layout is not swizzled, we can trivially vectorize stores
  // across the whole width of the most-minor dimension of the shape, because
  // Triton requires all the dims are powers of 2.
  unsigned outVec = dstSharedLayout.getMaxPhase() == 1
                        ? dstTy.getShape()[inOrd[0]]
                        : dstSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  auto wordTy = vec_ty(elemTy, minVec);
  Value word;

  SmallVector<Value, 3> srcStrides(dstStrides);
  SmallVector<Value, 3> offsetVals(rank, i32_val(0));
  SharedMemoryObject smemObj(smemBase, elemTy, srcStrides, offsetVals);

  DenseMap<unsigned, Value> sharedPtrs =
      getSwizzledSharedPtrs(loc, target, inVec, srcTy, dstSharedLayout, elemTy,
                            smemObj, rewriter, offsetVals, srcStrides);
  LDBG("storeDistributedToShared: numElems = " << numElems << " minVec = "
                                               << minVec << " " << wordTy);
  for (unsigned i = 0; i < numElems; ++i) {
    if (i % minVec == 0)
      word = undef(wordTy);
    word = insert_element(wordTy, word, inVals[i], i32_val(i % minVec));
    if (i % minVec == minVec - 1) {
      Value smemAddr = sharedPtrs[i / minVec * minVec];
    #ifndef USE_MACA_OPAQUE_PTR
      smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), wordTy, 3));
    #else
      smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3)); 
    #endif
      store(word, smemAddr)
          .setAlignment(minVec * elemTy.getIntOrFloatBitWidth() / 8);
    }
  }
}

inline Value
getStructFromSharedMemoryObject(Location loc, const SharedMemoryObject &smemObj,
                                ConversionPatternRewriter &rewriter) {
#ifdef USE_MACA
  auto elems = smemObj.getElems(loc, rewriter);
#else
  auto elems = smemObj.getElems();
#endif
  auto types = smemObj.getTypes();
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  // pack into struct
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
  for (const auto &v : llvm::enumerate(elems)) {
    assert(v.value() && "can not insert null values");
    llvmStruct = insert_val(structTy, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

inline SmallVector<Value>
unpackLLElements(Location loc, Value llvmStruct,
                 ConversionPatternRewriter &rewriter) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      isa<triton::PointerType>(llvmStruct.getType()) ||
      isa<LLVM::LLVMPointerType>(llvmStruct.getType()))
    return {llvmStruct};
  ArrayRef<Type> types =
      cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> results(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = extract_val(type, llvmStruct, i);
  }
  return results;
}

inline Value packLLElements(Location loc,
                            const LLVMTypeConverter *typeConverter,
                            ValueRange resultVals,
                            ConversionPatternRewriter &rewriter, Type type) {
  auto structType =
      dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
  if (!structType) {
    assert(resultVals.size() == 1);
    return *resultVals.begin();
  }

  auto elementTypes = structType.getBody();
  if (elementTypes.size() != resultVals.size()) {
    emitError(loc) << " size mismatch when packing elements for LLVM struct"
                   << " expected " << elementTypes.size() << " but got "
                   << resultVals.size();
  }
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  for (const auto &v : llvm::enumerate(resultVals)) {
    if (!v.value()) {
      emitError(loc)
          << "cannot insert null values into struct, but tried to insert"
          << v.value();
    }
    if (v.value().getType() != elementTypes[v.index()]) {
      LDBG("type " << type << " structType " << structType);
      LDBG("value " << v.value());
      emitError(loc) << "invalid element type in packLLEElements. Expected "
                     << elementTypes[v.index()] << " but got "
                     << v.value().getType();
    }
    llvmStruct = insert_val(structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

inline bool isLayoutMmaV1(Attribute layout) {
  bool isMmaV1 = false;
  if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
    isMmaV1 = mmaLayout.isVolta();
  }
  if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
    isMmaV1 = isa<NvidiaMmaEncodingAttr>(sliceLayout.getParent()) &&
              cast<NvidiaMmaEncodingAttr>(sliceLayout.getParent()).isVolta();
  }
  return isMmaV1;
}

inline bool isLayoutMmaMACA(Attribute layout) {
  if (auto mmaLayout = dyn_cast<MACAMmaEncodingAttr>(layout)) {
    return true;
  }
  if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
    return isa<MACAMmaEncodingAttr>(sliceLayout.getParent());
  }
  return false;
}

} // namespace mlir

#endif
