#include "Utility.h"
#include "mlir/Dialect/LLVMIR/MACADialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace LLVM {
namespace METAX {
using namespace mlir::triton;

// static Value shuffleCommon(Location loc, ConversionPatternRewriter &rewriter,
//                            Value val, Value i, NVVM::ShflKind mode,
//                            Value clamp) {
//   unsigned bits = val.getType().getIntOrFloatBitWidth();

//   if (bits == 64) {
//     Type vecTy = vec_ty(f32_ty, 2);
//     Value vec = bitcast(val, vecTy);
//     Value val0 = extract_element(f32_ty, vec, i32_val(0));
//     Value val1 = extract_element(f32_ty, vec, i32_val(1));
//     val0 = shuffleCommon(loc, rewriter, val0, i, mode, clamp);
//     val1 = shuffleCommon(loc, rewriter, val1, i, mode, clamp);
//     vec = undef(vecTy);
//     vec = insert_element(vecTy, vec, val0, i32_val(0));
//     vec = insert_element(vecTy, vec, val1, i32_val(1));
//     return bitcast(vec, val.getType());
//   }
//   Type type = val.getType();
//   if (type != i32_ty) {
//     val = bitcast(val, int_ty(bits));
//     if (bits < 32)
//       val = zext(i32_ty, val);
//   }
//   Value mask = i32_val(0xFFFFFFFF);
//   Value result = rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, val, i,
//   clamp,
//                                                mode, UnitAttr());
//   if (type != i32_ty) {
//     if (bits < 32)
//       result = trunc(int_ty(bits), result);
//     result = bitcast(result, type);
//   }
//   return result;
// }

Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i, Value tid, Operation *op) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();
  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleXor(loc, rewriter, val0, i, tid, op);
    val1 = shuffleXor(loc, rewriter, val1, i, tid, op);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }
  // #define BUILTIN_SHFL_XOR(MASK, VAR, LANE_DELTA, WIDTH) (                 \
  //     {                                                                    \
  //         int self = __get_lane_id();                                      \
  //         int index = self ^ LANE_DELTA;                                   \
  //         index = index >= ((self + WIDTH) & ~(WIDTH - 1)) ? self : index; \
  //         if (((1 << index) & MASK) == 0)                                  \
  //         {                                                                \
  //             tmp = VAR;                                                   \
  //         }                                                                \
  //         else                                                             \
  //         {                                                                \
  //             tmp = __builtin_mxc_bsm_bpermute(index << 2, VAR);           \
  //         }                                                                \
  //     })
  // maca shfl.sync.bfly, mask default to 0xffffffff in PTX, 'if (((1 << index)
  // & MASK) == 0)' is ignored
  // TODO:shfl.sync.down、shfl.sync.up、shfl.sync.idx in PTX and support mask
  // manual
  Value width = i32_val(64);       // set width to 64
  Value iValue = i32_val(i);       // delta
  Value index = xor_(tid, iValue); // tid ^ delta
  Value andValue = and_(add(tid, width), i32_val(-64));
  Value cmpValue = icmp_uge(index, andValue);       // index >= andValue
  Value selectValue = select(cmpValue, tid, index); // index>=andValue?tid:index
  Value shflValue = shl(selectValue, i32_val(2));   // selectValue << 2
  // TODO(): remove this if we support ::mxgpu::bpermute** etc in maca
  StringRef funcName("llvm.mxc.bsm.bpermute");
  if (bits == 32) {
    Value i32Val = bitcast(val, IntegerType::get(rewriter.getContext(), 32));
    ValueRange valueRange({shflValue, i32Val});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, i32Val.getType(),
                                 valueRange);
    return bitcast(ret, val.getType());
  } else {
    Value iVal = bitcast(val, IntegerType::get(rewriter.getContext(), bits));
    Value zextVal = zext(IntegerType::get(rewriter.getContext(), 32), iVal);
    ValueRange valueRange({shflValue, zextVal});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, zextVal.getType(),
                                 valueRange);
    auto truncRet = trunc(IntegerType::get(rewriter.getContext(), bits), ret);
    return bitcast(truncRet, val.getType());
  }
}

Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i, Value laneId, Operation *op) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleUp(loc, rewriter, val0, i, laneId, op);
    val1 = shuffleUp(loc, rewriter, val1, i, laneId, op);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

  //   _device__ MACA_INLINE int __shfl_up_sync(unsigned long mask, int var,
  //                                           unsigned int laneDelta,
  //                                           int width = warpSize) {
  //   int self = __lane_id();
  //   int index = self - laneDelta;
  //   index = (index < (self & ~(width - 1))) ? self : index;
  //   return __builtin_mxc_bsm_bpermute(index << 2, var);
  // }
  Value warpSize = i32_val(64);
  Value self = laneId;
  Value iValue = i32_val(i); // laneDelta
  Value index = sub(self, iValue);
  Value andValue = and_(self, i32_val(-64));
  Value cmpValue = icmp_ult(index, andValue); // (index < (self & ~(width - 1)))
  Value selectValue = select(
      cmpValue, self, index); // (index < (self & ~(width - 1))) ? self : index
  Value shflValue = shl(selectValue, i32_val(2)); // selectValue << 2
  StringRef funcName("llvm.mxc.bsm.bpermute");
  if (bits == 32) {
    Value i32Val = bitcast(val, IntegerType::get(rewriter.getContext(), 32));
    ValueRange valueRange({shflValue, i32Val});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, i32Val.getType(),
                                 valueRange);
    return bitcast(ret, val.getType());
  } else {
    Value iVal = bitcast(val, IntegerType::get(rewriter.getContext(), bits));
    Value zextVal = zext(IntegerType::get(rewriter.getContext(), 32), iVal);
    ValueRange valueRange({shflValue, zextVal});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, zextVal.getType(),
                                 valueRange);
    auto truncRet = trunc(IntegerType::get(rewriter.getContext(), bits), ret);
    return bitcast(truncRet, val.getType());
  }
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i, Operation *op) {
  return shuffleIdx(loc, rewriter, val, i32_val(i), op);
}

Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i, Operation *op) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleIdx(loc, rewriter, val0, i, op);
    val1 = shuffleIdx(loc, rewriter, val1, i, op);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }
  Value shflValue = shl(i, i32_val(2)); // selectValue << 2
  StringRef funcName("llvm.mxc.bsm.bpermute");
  if (bits == 32) {
    Value i32Val = bitcast(val, IntegerType::get(rewriter.getContext(), 32));
    ValueRange valueRange({shflValue, i32Val});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, i32Val.getType(),
                                 valueRange);
    return bitcast(ret, val.getType());
  } else {
    Value iVal = bitcast(val, IntegerType::get(rewriter.getContext(), bits));
    Value zextVal = zext(IntegerType::get(rewriter.getContext(), 32), iVal);
    ValueRange valueRange({shflValue, zextVal});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, zextVal.getType(),
                                 valueRange);
    auto truncRet = trunc(IntegerType::get(rewriter.getContext(), bits), ret);
    return bitcast(truncRet, val.getType());
  }
}

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
  Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr) {
  // PTXBuilder builder;
  // auto &mov = builder.create("mov")->o("u32");
  // auto *destOpr = builder.newOperand("=r");
  // auto *sRegOpr = builder.newConstantOperand(sRegStr);
  // mov(destOpr, sRegOpr);
  // Value val = builder.launch(b, loc, b.getIntegerType(32), false);
  Value val;
  return val;
}

Value permute(Location loc, ConversionPatternRewriter &rewriter, Value a,
              Value b, Value mask) {
  // PTXBuilder builder;
  // auto &prmt = builder.create("prmt")->o("b32");
  // auto *destOpr = builder.newOperand("=r");
  // auto *aOperand = builder.newOperand(a, "r");
  // auto *bOperand = builder.newOperand(b, "r");
  // auto *maskOperand = builder.newOperand(mask, "r");
  // prmt(destOpr, aOperand, bOperand, maskOperand);
  // return builder.launch(rewriter, loc, rewriter.getIntegerType(32), false);
  Value result;
  return result;
}

// A wrapper of LoadDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp
// (3) Bitcast result from dataTy (u16/u32/u64) back to elemTy
Value createLoadDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Type elemTy) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  // Value ret =
  //     rewriter.create<triton::nvgpu::LoadDSmemOp>(loc, addr, ctaId,
  //     bitwidth);
  Value ret;
  return bitcast(ret, elemTy);
}

// A wrapper of LoadDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp and extract results from retStruct
// (3) Bitcast results from dataTy (u16/u32/u64) back to elemTy
SmallVector<Value> createLoadDSmem(Location loc, PatternRewriter &rewriter,
                                   Value addr, Value ctaId, unsigned vec,
                                   Type elemTy) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  // Value retStruct = rewriter.create<triton::nvgpu::LoadDSmemOp>(
  //     loc, addr, ctaId, bitwidth, vec);
  SmallVector<Value> retVals;
  // for (unsigned i = 0; i < vec; ++i) {
  //   auto dataTy = rewriter.getIntegerType(bitwidth);
  //   Value data = extract_val(dataTy, retStruct, i);
  //   retVals.push_back(bitcast(data, elemTy));
  // }
  return retVals;
}

// A wrapper of StoreDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Bitcast value from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value, Value pred) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = value.getType().getIntOrFloatBitWidth();
  auto dataTy = rewriter.getIntegerType(bitwidth);
  Value data = bitcast(value, dataTy);
  // rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec = 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, value, pred);
}

// A wrapper of StoreDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Bitcast values from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values, Value pred) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = 0;
  if (!values.empty()) {
    bitwidth = values.back().getType().getIntOrFloatBitWidth();
  }
  auto dataTy = rewriter.getIntegerType(bitwidth);
  SmallVector<Value> data;
  for (unsigned i = 0; i < values.size(); ++i)
    data.push_back(bitcast(values[i], dataTy));
  // rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec > 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, values, pred);
}

/// Create a predicate with just single active thread.
// Value createElectPredicate(Location loc, PatternRewriter &rewriter) {
//   PTXBuilder ptxBuilder;
//   auto &elect = *ptxBuilder.create<>("elect.sync _|$0, 0xffffffff;");
//   elect({ptxBuilder.newOperand("=b")}, /*onlyAttachMLIRArgs=*/true);
//   // The instruction is technically not pure as it depends on simt control
//   flow
//   // however since we it outside of simt control flow in triton we can
//   consider
//   // it as pure to allow cse to work on it.
//   return ptxBuilder.launch(rewriter, loc, i1_ty, /*hasSideEffect=*/false);
// }
Type getFunctionType(Type resultType, const ValueRange &operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                   Operation *op, StringRef funcName,
                                   Type funcType) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(rewriter.getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  mlir::OpBuilder b(op->template getParentOfType<LLVMFuncOp>());
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  return ret;
}

__attribute__((optnone)) __attribute__((noinline)) Value createBuiltinFunc(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    StringRef funcName, Type resultType, const ValueRange &argValues) {
  Type funcType = getFunctionType(resultType, argValues);
  auto funcOp = appendOrGetFuncOp(rewriter, op, funcName, funcType);
  return rewriter.create<LLVM::CallOp>(loc, funcOp, argValues).getResult();
}

} // namespace METAX

Type getFunctionType(Type resultType, const ValueRange &operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

} // namespace LLVM
} // namespace mlir

namespace maca {
bool get_triton_disable_fp32_to_bf16_high_precsion() {
  static char const *temp =
      std::getenv("TRITON_DISABLE_FP32_TO_BF16_HIGH_PRECISION");
  return temp != nullptr;
}

} // namespace maca
