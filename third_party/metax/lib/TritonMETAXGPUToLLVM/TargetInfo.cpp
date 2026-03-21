#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/MACADialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

using mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::LLVM::METAX::createBuiltinFunc;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
namespace {
Value computeStMatrixAddr(Value laneId, int matStride, Location loc,
                          ConversionPatternRewriter &rewriter,
                          int swizzleByteWidth) {
  Value rowInMat = urem(laneId, i32_val(8)); // row in the 8x8 matrix
  // linear index of the matrix in the 2x2 matrices
  // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in
  // a warp.
  Value matIndex = udiv(laneId, i32_val(8));
  Value s0 = urem(matIndex, i32_val(2));
  Value s1 = udiv(matIndex, i32_val(2));
  if (swizzleByteWidth >= 32)
    s1 = xor_(s1, and_(laneId, i32_val(1)));
  Value mIndex = add(rowInMat, mul(s0, i32_val(8)));
  int m8n8Stride = 8;
  Value offset =
      add(mul(mIndex, i32_val(matStride)), mul(s1, i32_val(m8n8Stride)));
  return offset;
}

void stMatrixm8n8x4(Value offset, ArrayRef<Value> vals, int indexOffset,
                    Value smemBase, Type elemTy, Location loc,
                    ConversionPatternRewriter &rewriter) {
  SmallVector<Value> inputs;
  // auto prTy = ptr_ty(rewriter.getContext(), 3);
  //  Pack the input into 2xf16
  Type packedTy = vec_ty(vals[0].getType(), 2);
  for (int i = 0; i < 4; i++) {
    Value input = undef(packedTy);
    for (int j = 0; j < 2; j++) {
      input = insert_element(packedTy, input, vals[indexOffset + i * 2 + j],
                             i32_val(j));
    }
    inputs.push_back(bitcast(input, i32_ty));
  }
  Value addr = gep(smemBase.getType(), elemTy, smemBase, offset);
  // rewriter.create<triton::nvgpu::StoreMatrixOp>(loc, addr, inputs);
}
void storeDistributedToSharedWithStMatrix(
    RankedTensorType tensorTy, Type elemTy, SmallVector<Value> &inVals,
    Value smemBase, ArrayRef<unsigned> paddedRepShape,
    ArrayRef<unsigned> origRepShape, Location loc,
    ConversionPatternRewriter &rewriter, int swizzlingByteWidth) {
  llvm_unreachable("Not support");
}

bool isStMatrixCompatible(RankedTensorType tensorTy, int swizzlingByteWidth) {
  return false;
}

// declare vprintf(i8*, i8*) as external function
LLVM::LLVMFuncOp getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("vprintf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType, true);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                           funcType);
}

// extend integer to int32, extend float to float64
// this comes from vprintf alignment requirements.
std::pair<Type, Value> printfPromoteValue(ConversionPatternRewriter &rewriter,
                                          Value value) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  Value newOp = value;
  Type newType = type;
  auto loc = UnknownLoc::get(context);

  bool isUnsigned = type.isUnsignedInteger();
  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    if (isUnsigned) {
      newType = ui32_ty;
      newOp = zext(newType, value);
    } else {
      newType = i32_ty;
      newOp = sext(newType, value);
    }
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    newType = f64_ty;
    newOp = fpext(newType, value);
  }

  return {newType, newOp};
}

LLVM::LLVMFuncOp getVprintfDeclaration(ConversionPatternRewriter &rewriter,
                                       ValueRange args) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();

  StringRef funcName("llvm.mxc.vprintf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  Type newType;
  Value newArg;

#ifndef USE_MACA_OPAQUE_PTR
  SmallVector<Type> argsType{ptr_ty(i8_ty)};
#else
  SmallVector<Type> argsType{ptr_ty(context)};
#endif
  // for (int i = 0; i < args.size(); i++) {
  //   std::tie(newType, newArg) = printfPromoteValue(rewriter, args[i]);
  //   argsType.push_back(newType);
  // }

  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType, true);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto *ctx = rewriter.getContext();
  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                           funcType);
}

LLVM::LLVMFuncOp getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("llvm.mxc.assertfail");
  {
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);
  }
  // void __assert_fail(const char * assertion, const char * file, unsigned
  // int line, const char * function);
  auto *ctx = rewriter.getContext();
#ifndef USE_MACA_OPAQUE_PTR
  SmallVector<Type> argsType{ptr_ty(i8_ty), ptr_ty(i8_ty), i64_ty,
                             ptr_ty(i8_ty)};
#else
  SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i64_ty, ptr_ty(ctx)};
#endif
  auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);
  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx),
                                                  funcName, funcType);

  funcOp.setPassthroughAttr(
      ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
  return funcOp;
}
} // namespace

namespace mlir::triton::METAX {

// Check if the reduction can use a redux op and return the kind.
// static std::optional<NVVM::ReduxKind> matchReduxKind(triton::ReduceOp op,
//                                                      int computeCapability) {
//   if (computeCapability < 80)
//     return std::nullopt;
//   if (op.getNumOperands() != 1 || op.getNumResults() != 1)
//     return std::nullopt;
//   Block *block = &(*op.getCombineOp().begin());
//   Operation *yield = block->getTerminator();
//   Operation *reduceOp = yield->getOperand(0).getDefiningOp();
//   if (!reduceOp || reduceOp->getNumOperands() != 2 ||
//       reduceOp->getNumResults() != 1)
//     return std::nullopt;
//   auto intType = dyn_cast<IntegerType>(reduceOp->getResultTypes()[0]);
//   if (!intType || intType.getWidth() > 32)
//     return std::nullopt;
//   if (reduceOp->getOperand(0) != block->getArgument(0) ||
//       reduceOp->getOperand(1) != block->getArgument(1))
//     return std::nullopt;
//   // TODO: jcheng
//   // if (isa<arith::AddIOp>(reduceOp))
//   //   return NVVM::ReduxKind::ADD;
//   // if (isa<arith::AndIOp>(reduceOp))
//   //   return NVVM::ReduxKind::AND;
//   // if (isa<arith::OrIOp>(reduceOp))
//   //   return NVVM::ReduxKind::OR;
//   // if (isa<arith::XOrIOp>(reduceOp))
//   //   return NVVM::ReduxKind::XOR;
//   // if (isa<arith::MinSIOp>(reduceOp))
//   //   return NVVM::ReduxKind::MIN;
//   // if (isa<arith::MinUIOp>(reduceOp))
//   //   return NVVM::ReduxKind::UMIN;
//   // if (isa<arith::MaxSIOp>(reduceOp))
//   //   return NVVM::ReduxKind::MAX;
//   // if (isa<arith::MaxUIOp>(reduceOp))
//   //   return NVVM::ReduxKind::UMAX;
//   return std::nullopt;
// }

bool TargetInfo::supportMaximumMinimum() const {
  // maximum / minimum nan compare unsupported
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
// TODO: MACA intrinsic builder requires operation to look up nearest symbol
// So we have to add op to the common interface.
#ifdef USE_MACA
                         Type type, Value cmp, Operation *op) const {
#else
                         Type type, Value cmp) const {
#endif
  Value threadMask = int_val(type.getIntOrFloatBitWidth(), -1);
#ifdef USE_LLVM19_INTRINSIC
  StringRef funcName("llvm.mxc.icmp.i64.i32");
#else
  StringRef funcName("llvm.mxc.icmp.i32");
#endif
  Value icmpNeVal = i32_val(33);
  Value zeroVal = i32_val(0);
  Value cmpValue = zext(i32_ty, cmp);
  ValueRange valueRange({cmpValue, zeroVal, icmpNeVal});
  auto ret = createBuiltinFunc(rewriter, loc, op, funcName, type, valueRange);
  return ret;
}

void TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Value val, Value pred) const {
  rewriter.create<scf::IfOp>(
      loc, pred,
      [&](OpBuilder &builder, Location loc) {
        store(val, ptr);
        builder.create<mlir::scf::YieldOp>(loc, ValueRange({}));
      },
      [&](OpBuilder &builder, Location loc) {
        builder.create<mlir::scf::YieldOp>(loc, ValueRange({}));
      });
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             const TypeConverter *converter, Value ptr,
                             Type elemTy, Value pred) const {
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for loadShared");
  unsigned bitwidth = std::max(8u, elemTy.getIntOrFloatBitWidth());

  const char *c = bitwidth == 64 ? "=l" : (bitwidth == 16 ? "=h" : "=r");

  // PTXBuilder builder;
  // auto *dOpr = builder.newOperand(c);
  // auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  // auto &ld = builder.create<>("ld")->shared().b(bitwidth);
  // ld(dOpr, ptrOpr).predicate(pred, "b");
  // return builder.launch(rewriter, loc, elemTy);
  Value zeroVal = int_val(bitwidth, 0);
  if (bitwidth == elemTy.getIntOrFloatBitWidth()) {
    zeroVal = bitcast(zeroVal, elemTy);
  } else {
    zeroVal = int_val(elemTy.getIntOrFloatBitWidth(), 0);
  }
  auto loaded = rewriter.create<scf::IfOp>(
      loc, pred,
      [&](OpBuilder &builder, Location loc) {
        auto loadVal = load(elemTy, ptr);
        builder.create<mlir::scf::YieldOp>(loc, ValueRange({loadVal}));
      },
      [&](OpBuilder &builder, Location loc) {
        Value otherVal = zeroVal;
        builder.create<mlir::scf::YieldOp>(loc, ValueRange({otherVal}));
      });
  return loaded->getResult(0);
}

Value TargetInfo::shuffleXor(ConversionPatternRewriter &rewriter, Location loc,
// TODO: MACA intrinsic builder requires operation to look up nearest symbol
// So we have to add op to the common interface.
#ifdef USE_MACA
                             Value val, int i, Operation *op) const {
#else
                             Value val, int i) const {
#endif
  Value threadId = getThreadId(rewriter, loc);
#ifdef USE_MACA
  return LLVM::METAX::shuffleXor(loc, rewriter, val, i, threadId, op);
#else
  return LLVM::METAX::shuffleXor(loc, rewriter, val, i, threadId);
#endif
}

Value TargetInfo::shuffleUp(ConversionPatternRewriter &rewriter, Location loc,
// TODO: MACA intrinsic builder requires operation to look up nearest symbol
// So we have to add op to the common interface.
#ifdef USE_MACA
                            Value val, int i, Operation *op) const {
#else
                            Value val, int i) const {
#endif
  Value threadId = getThreadId(rewriter, loc);
#ifdef USE_MACA
  return LLVM::METAX::shuffleUp(loc, rewriter, val, i, threadId, op);
#else
  return LLVM::METAX::shuffleUp(loc, rewriter, val, i, threadId);
#endif
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
#ifdef USE_MACA
                             Value val, int i, Operation *op) const {
  return LLVM::METAX::shuffleIdx(loc, rewriter, val, i, op);
#else
                             Value val, int i) const {
  return LLVM::METAX::shuffleIdx(loc, rewriter, val, i);
#endif
}

Value TargetInfo::shuffleIdx(ConversionPatternRewriter &rewriter, Location loc,
#ifdef USE_MACA
                             Value val, Value i, Operation *op) const {
  return LLVM::METAX::shuffleIdx(loc, rewriter, val, i, op);
#else
                             Value val, Value i) const {
  return LLVM::METAX::shuffleIdx(loc, rewriter, val, i);
#endif
}

Value TargetInfo::programId(ConversionPatternRewriter &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::METAX::llGetPid(loc, rewriter, moduleOp, axis);
}
bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  // Value threadId = getThreadId(rewriter, loc);
  // for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
  //   SmallVector<Value> shfl(acc.size());
  //   for (unsigned i = 0; i < acc.size(); ++i) {
  //       shfl[i] = LLVM::METAX::shuffleXor(loc, rewriter, acc[i], N, threadId,
  //       op);
  //   }
  //   accumulate(rewriter, op.getCombineOp(), acc, shfl, false);
  // }
  // return true;
  return false;
}
bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
    int swizzlingByteWidth) const {
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName = resultElementTy.isInteger(32)
                             ? "mc_math_func_umulhi"
                             : "mc_math_func_umul64hi";
  return funcName;
}

void TargetInfo::printf(ConversionPatternRewriter &rewriter,
                        Value formatStrStart, int /*formatStrByteCount*/,
                        ValueRange args) const {
  auto *ctx = rewriter.getContext();
#ifndef USE_MACA_OPAQUE_PTR
  Type ptr = ptr_ty(i8_ty);
#else
  Type ptr = ptr_ty(ctx);
#endif
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getVprintfDeclaration(rewriter, args);
  auto loc = UnknownLoc::get(ctx);

  Value one = i32_val(1);
  Value zero = i32_val(0);

  Value bufferPtr = null(ptr);

  SmallVector<Value> operands{formatStrStart};
  for (int i = 0; i < args.size(); i++) {
    Type newType;
    Value newArg;
    std::tie(newType, newArg) = printfPromoteValue(rewriter, args[i]);
    operands.push_back(newArg);
  }
  call(funcOp, operands);
}

void TargetInfo::assertFail(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  std::string msg = message.str() + "\00";
  size_t msg_len = message.size();
  StringRef msg_ = StringRef(msg.data(), msg_len + 1);
  Value messageString =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", msg_);
  std::string f = file.str() + "\00";
  size_t f_len = file.size();
  StringRef f_ = StringRef(file.data(), f_len + 1);
  Value fileString = LLVM::addStringToModule(loc, rewriter, "assertFile_", f_);
  std::string fun = func.str() + "\00";
  size_t fun_len = fun.size();
  StringRef fun_ = StringRef(func.data(), fun_len + 1);
  Value funcString =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", fun_);
  Value lineNumber = i32_val(line);
  Value lineN = sext(IntegerType::get(rewriter.getContext(), 64), lineNumber);
  SmallVector<Value> operands = {messageString, fileString, lineN, funcString};
  call(funcOp, operands);
}

} // namespace mlir::triton::METAX
