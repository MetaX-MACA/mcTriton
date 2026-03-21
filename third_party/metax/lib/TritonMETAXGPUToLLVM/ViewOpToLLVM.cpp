#include "TargetInfo.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct ExtractTensorOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ExtractTensorOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::ExtractTensorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExtractTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_tensor %source [%tileIdx][%subsizes]
    Location loc = op->getLoc();
    auto srcTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    if (!srcTy)
      return failure();
    auto srcLayout = srcTy.getEncoding();
    Value result = op.getResult();
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      return failure();
    auto ctaIdx = op.getCtaIdx();
    auto elemIdx = op.getElemIdx();
    SmallVector<Value> subelems;
    ArrayRef<Type> types =
        cast<LLVM::LLVMStructType>(adaptor.getSource().getType()).getBody();
    auto subIdx = emitSubOffsetForLayout(srcLayout, srcTy, ctaIdx, elemIdx);
    for (unsigned i : subIdx) {
      subelems.push_back(extract_val(types[i], adaptor.getSource(), i));
    }
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), subelems, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct InsertTensorOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::InsertTensorOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::InsertTensorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::InsertTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = insert_tensor %inserted %insert [%tileIdx][%subsizes]
    Location loc = op->getLoc();
    auto insertedTy = dyn_cast<RankedTensorType>(op.getInserted().getType());
    if (!insertedTy)
      return failure();
    auto insertedLayout = insertedTy.getEncoding();
    auto insertTy = dyn_cast<RankedTensorType>(op.getInsert().getType());
    if (!insertTy)
      return failure();
    Value result = op.getResult();
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      return failure();
    auto ctaIdx = op.getCtaIdx();
    auto elemIdx = op.getElemIdx();
    auto subelems = unpackLLElements(loc, adaptor.getInsert(), rewriter);
    auto subIdx =
        emitSubOffsetForLayout(insertedLayout, insertedTy, ctaIdx, elemIdx);
    unsigned idx = 0;
    auto resultStruct = adaptor.getInserted();
    auto resultStructTy = dyn_cast<LLVM::LLVMStructType>(
        getTypeConverter()->convertType(resultTy));
    assert(subIdx.size() == subelems.size());
    for (unsigned i : subIdx) {
      resultStruct = insert_val(resultStructTy, resultStruct, subelems[idx], i);
      ++idx;
    }
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BsmPermOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BsmPermOp> {
  using ConvertOpToLLVMPattern<triton::gpu::BsmPermOp>::ConvertOpToLLVMPattern;
  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  LogicalResult
  matchAndRewrite(triton::gpu::BsmPermOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = insert_tensor %inserted %insert [%tileIdx][%subsizes]
    Location loc = op->getLoc();
    auto ctx = rewriter.getContext();
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc1().getType());
    auto srcLayout = srcTy.getEncoding();
    SmallVector<unsigned> order{1, 0};
    Value result = op.getResult();
    auto resultTy = cast<RankedTensorType>(result.getType());
    ArrayRef<Type> types =
        cast<LLVM::LLVMStructType>(adaptor.getSrc1().getType()).getBody();

    auto mmaEnc = dyn_cast<MACAMmaEncodingAttr>(
        dyn_cast<DotOperandEncodingAttr>(resultTy.getEncoding()).getParent());

    auto elemsMnk = mmaEnc.getElementsMNK();
    unsigned tm = elemsMnk[0];
    unsigned tn = elemsMnk[1];
    unsigned tk = elemsMnk[2];
    SmallVector<unsigned> elemsPerThread(2);
    elemsPerThread[0] = tk;
    elemsPerThread[1] = tn;
    SmallVector<Value> subelems(tk * tn);

    Type typesrc = srcTy.getElementType();
    Type elemType;
    if (resultTy.getElementType().isF16()) {
      elemType = type::f16Ty(ctx);
    } else if (resultTy.getElementType().isBF16()) {
      elemType = type::i16Ty(ctx);
    } else if (resultTy.getElementType().isF32()) { // TF32
      elemType = type::f32Ty(ctx);
    } else {
      assert(false && "Invalid smem load");
    }

    Type fp16x2Ty = vec_ty(elemType, 2);
    Value halfValuex2Front = undef(fp16x2Ty);
    Value halfValuex2Back = undef(fp16x2Ty);
    Type int32Ty = type::i32Ty(ctx);

    int kOrder = 0;
    auto ld2Opt = [&](ValueTable &vals, int mn, int k,
                      SmallVector<Value> val_vec) {
      for (int j = 0; j < elemsPerThread[order[1]]; ++j) {
        for (int i = 0; i < elemsPerThread[order[0]]; ++i) {
          if (kOrder != order[0]) { // a [0,1], b [1,0]
            vals[{i, j}] = val_vec[j * elemsPerThread[order[0]] + i];
          } else { // a [1,0], b[0,1]
            vals[{mn * elemsPerThread[order[1]] + j,
                  k * elemsPerThread[order[0]] + i}] =
                val_vec[j * elemsPerThread[order[0]] + i];
          }
        }
      }
    };

    Value offsetFront = i32_val(0x01000504);
    Value offsetBack = i32_val(0x03020706);

    SmallVector<Value> rangeValueFront(3);
    rangeValueFront[2] = offsetFront;
    SmallVector<Value> rangeValueBack(3);
    rangeValueBack[2] = offsetBack;

    Value permValueInt32Front = undef(i32_ty);
    Value permValueInt32Back = undef(i32_ty);
    std::string intrinsicPermName = "llvm.mxc.byte.perm";
    StringRef permName(intrinsicPermName);

    for (int j = 0; j < tk; j += 2) {
      for (int idx = 0; idx < tn / 2; ++idx) {
        unsigned index = 4 * idx + j * tn;
        rangeValueFront[0] =
            extract_val(types[index], adaptor.getSrc1(), index);
        rangeValueFront[1] =
            extract_val(types[index + 1], adaptor.getSrc1(), index + 1);
        ValueRange permValueRangeFront(rangeValueFront);

        rangeValueBack[0] =
            extract_val(types[index + 2], adaptor.getSrc1(), index + 2);
        rangeValueBack[1] =
            extract_val(types[index + 3], adaptor.getSrc1(), index + 3);
        ValueRange permValueRangeBack(rangeValueBack);

        permValueInt32Front = mlir::LLVM::createBuiltinFunc(
            rewriter, loc, op, permName, int32Ty, permValueRangeFront);
        halfValuex2Front = bitcast(permValueInt32Front, fp16x2Ty);

        permValueInt32Back = mlir::LLVM::createBuiltinFunc(
            rewriter, loc, op, permName, int32Ty, permValueRangeBack);
        halfValuex2Back = bitcast(permValueInt32Back, fp16x2Ty);

        subelems[2 * idx + j * tn] =
            extract_element(halfValuex2Front, i32_val(0));
        subelems[2 * idx + (j + 1) * tn] =
            extract_element(halfValuex2Front, i32_val(1));

        subelems[2 * idx + 1 + j * tn] =
            extract_element(halfValuex2Back, i32_val(0));
        subelems[2 * idx + 1 + (j + 1) * tn] =
            extract_element(halfValuex2Back, i32_val(1));
      }
    }

    ValueTable vals;
    ld2Opt(vals, 0, 0, subelems);

    auto wpts = mmaEnc.getWarpsPerCTA();
    auto tensorTy = cast<TensorOrMemDesc>(srcTy);
    SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                               tensorTy.getShape().end());
    int numRepK = mlir::triton::gpu::getNumRepK(shape[0], elemsPerThread[0]);
    int numRepN =
        mlir::triton::gpu::getNumRepN(shape[1], wpts, elemsPerThread[1]);
    std::vector<Value> elems;
    for (int n = 0; n < std::max(numRepN, 1); ++n)
      for (int k = 0; k < numRepK; ++k)
        for (int j = 0; j < tn; ++j)
          for (int i = 0; i < tk; ++i) {
            elems.push_back(vals.at({n * tn + j, k * tk + i}));
          }

    Type elemTy = elems[0].getType();
    Type structTy = LLVM::LLVMStructType::getLiteral(
        elemTy.getContext(), SmallVector<Type>(elems.size(), elemTy));
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), elems, rewriter, structTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

inline Value getCopyAsyncSwizzledMask_1(
    Location loc, ConversionPatternRewriter &rewriter,
    const TargetInfoBase &target, unsigned inVec, unsigned outVec,
    unsigned perPhase, unsigned maxPhase, RankedTensorType srcTy,
    RankedTensorType resTy, unsigned elemIdx, Value startMaskPerLine) {
  unsigned minVec = std::min(inVec, outVec);
  auto srcEncoding = srcTy.getEncoding();

  // order
  auto outOrder = triton::gpu::getOrder(resTy.getEncoding());

  // tensor indices held by the current thread, as LLVM values
  auto srcIndices = emitIndices(loc, rewriter, target, srcEncoding, srcTy,
                                /*withCTAOffset=*/false);

  // extract multi dimensional index for current element
  auto idx = srcIndices[elemIdx];
  Value idxCol = idx[outOrder[0]]; // contiguous dimension
  Value idxRow = idx[outOrder[1]]; // discontiguous dimension

  if (maxPhase == 1) {
    return startMaskPerLine;
  } else if (outVec == minVec) {
    Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
    Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
    colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
    Value colOff = colOffSwizzled;

    auto elemType = startMaskPerLine.getType();
    if (elemType.isInteger(64)) {
      colOff = rewriter.create<arith::ExtSIOp>(loc, elemType, colOff);
    }
    Value currMask = add(startMaskPerLine, colOff);
    return currMask;
  } else {
    Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
    Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
    colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
    Value colOffOrdered = urem(idxCol, i32_val(outVec));
    colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
    colOffOrdered = mul(colOffOrdered, i32_val(minVec));
    Value colOff = add(colOffSwizzled, colOffOrdered);
    auto elemType = startMaskPerLine.getType();
    if (elemType.isInteger(64)) {
      colOff = rewriter.create<arith::ExtSIOp>(loc, elemType, colOff);
    }
    Value currMask = add(startMaskPerLine, colOff);
    return currMask;
  }
}

struct SwizzleTensorOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::SwizzleTensorOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::SwizzleTensorOp>::ConvertOpToLLVMPattern;

  SwizzleTensorOpConversion(const LLVMTypeConverter &typeConverter,
                            const METAX::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::SwizzleTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_tensor %source [%tileIdx][%subsizes]
    Location loc = op->getLoc();
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy)
      return failure();
    auto maxPhase = op.getMaxPhase();
    if (maxPhase == 1) {
      rewriter.replaceOp(op, {adaptor.getSrc()});
      return success();
    }
    auto srcLayout = srcTy.getEncoding();
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcLayout);
    auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
    auto inOrder = triton::gpu::getOrder(srcLayout);
    assert(inOrder.size() == 2 && threadsPerWarp.size() == 2 &&
           "Unexpected Order rank of getContiguityGvmStartAddr");
    int getContiguityDimTNum;
    for (int i = 0; i < inOrder.size(); i++) {
      if (inOrder[i] == 0) {
        getContiguityDimTNum = threadsPerWarp[i];
        break;
      }
    }
    auto sizePerThread = triton::gpu::getSizePerThread(srcLayout);
    int numElemsPerThread = 0;
    int getContiDimSizePerCTA = 0;
    for (int i = 0; i < inOrder.size(); i++) {
      if (inOrder[i] == 0) {
        numElemsPerThread = sizePerThread[i];
        getContiDimSizePerCTA = srcShapePerCTA[i];
        break;
      }
    }

    Value result = op.getResult();
    auto inVec = op.getInVec();
    auto outVec = op.getOutVec();
    auto perPhase = op.getPerPhase();
    auto minVec = std::min(inVec, outVec);
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      return failure();
    auto elems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    unsigned numElems = elems.size();
    SmallVector<Value> llResult = elems;

    int getContiDimWarpNum =
        getContiDimSizePerCTA / (numElemsPerThread * getContiguityDimTNum);
    getContiDimWarpNum = getContiDimWarpNum == 0 ? 1 : getContiDimWarpNum;
    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      auto elem = elems[elemIdx];
      Value contiDimTNum = i32_val(64 * getContiDimWarpNum);
      Value threadId = getThreadId(rewriter, loc);
      Value lane = urem(threadId, contiDimTNum);
      Value tidMulti =
          udiv(lane, i32_val(getContiguityDimTNum * getContiDimWarpNum));
      Value thread =
          mul(tidMulti, i32_val(getContiguityDimTNum * getContiDimWarpNum));
      Value startThread = sub(thread, lane);
      Value maskOffset = mul(startThread, i32_val(numElemsPerThread));
      auto elemType = elem.getType();
      if (elemType.isInteger(64)) {
        maskOffset = rewriter.create<arith::ExtSIOp>(loc, elemType, maskOffset);
      }
      Value startMaskPerLine = add(elem, maskOffset);

      // 3. compute swizzle for mask lhs.
      Value swiMask = getCopyAsyncSwizzledMask_1(
          loc, rewriter, targetInfo, inVec, outVec, perPhase, maxPhase, srcTy,
          resultTy, elemIdx, startMaskPerLine);
      llResult[elemIdx] = swiMask;
    }
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), llResult, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }

private:
  const METAX::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::METAX::populateViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<ExtractTensorOpConversion>(typeConverter, benefit);
  patterns.add<InsertTensorOpConversion>(typeConverter, benefit);
  patterns.add<BsmPermOpConversion>(typeConverter, benefit);
  patterns.add<SwizzleTensorOpConversion>(typeConverter, targetInfo, benefit);
}
