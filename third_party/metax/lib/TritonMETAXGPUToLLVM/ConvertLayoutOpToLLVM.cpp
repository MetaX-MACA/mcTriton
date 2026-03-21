#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/MACADialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using mlir::isLayoutMmaMACA;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace SharedToDotOperandMMAMACA {
Value convertLayout(int opIdx, triton::gpu::LocalLoadOp op,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread,
                    ArrayRef<unsigned> elemsPerThread,
                    METAX::IndexCacheInfoSm indexCacheInfoSm, bool enSmIdxCache,
                    bool enSmIndexOpt);
}

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &typeConverter,
                        METAX::IndexCacheInfoSm indexCacheInfoSm,
                        bool enSmIdxCache, bool enSmIndexOpt,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        indexCacheInfoSm(indexCacheInfoSm), enSmIdxCache(enSmIdxCache),
        enSmIndexOpt(enSmIndexOpt) {}
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<MACAMmaEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<BlockedEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOpFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  METAX::IndexCacheInfoSm indexCacheInfoSm;
  bool enSmIdxCache;
  bool enSmIndexOpt;

  LogicalResult
  lowerSharedToDotOpFMA(triton::gpu::LocalLoadOp op,
                        triton::gpu::LocalLoadOpAdaptor adaptor,
                        const LLVMTypeConverter *typeConverter,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType dstTy = op.getType();
    Attribute dstLayout = dstTy.getEncoding();
    auto dotLayout = cast<DotOperandEncodingAttr>(dstLayout);
    auto blockedLayout = cast<BlockedEncodingAttr>(
        cast<DotOperandEncodingAttr>(dstLayout).getParent());
    auto thread = getThreadId(rewriter, loc);
    Value res = SharedToDotOperandFMA::convertLayout(
        dotLayout.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout,
        thread, loc, getTypeConverter(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }

  // shared -> dot_operand if the result layout is mma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter, const MACAMmaEncodingAttr &mmaLayout,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dst = op.getResult();

    auto llvmElemTy =
        typeConverter->convertType(src.getType().getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;
    if (!isOuter) { // maca tensor core
      auto elemsPerThread = mmaLayout.getElementsMNK();

      // TODO: default not allowTF32, support in future.
      res = SharedToDotOperandMMAMACA::convertLayout(
          dotOperandLayout.getOpIdx(), op, rewriter, loc, src, dotOperandLayout,
          smemObj, typeConverter, tid_val(), elemsPerThread, indexCacheInfoSm,
          enSmIdxCache, enSmIndexOpt);
    } else {
      assert(false && "Unsupported mma layout found");
    }
    return res;
  };

  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto dstEnc = cast<DotOperandEncodingAttr>(op.getType().getEncoding());
    auto sharedLayout =
        cast<SharedEncodingAttr>(op.getSrc().getType().getEncoding());

    int K;
    if (dstEnc.getOpIdx() == 0) // $a
      K = op.getType().getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = op.getType().getShape()[sharedLayout.getOrder()[1]];
    bool isOuter = K == 1;
    auto mmaLayout = cast<MACAMmaEncodingAttr>(dstEnc.getParent());
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           mmaLayout, dstEnc, isOuter);
    int size = cast<LLVM::LLVMStructType>(res.getType()).getBody().size();

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(const LLVMTypeConverter &typeConverter,
                            const METAX::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      if (isLayoutMmaMACA(srcLayout) || isLayoutMmaMACA(dstLayout)) {
        return lowerDistributedToDistributed(op, adaptor, rewriter);
      }
    }
    if (isa<MACAMmaEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }

    return failure();
  }

private:
  SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       unsigned elemId, RankedTensorType type,
                                       ArrayRef<unsigned> multiDimCTAInRepId,
                                       ArrayRef<unsigned> shapePerCTA,
                                       unsigned byteWidth = 4) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
      auto multiDimOffsetFirstElem = emitBaseIndexForLayout(
          loc, rewriter, targetInfo, blockedLayout, type, false);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] = add(multiDimOffsetFirstElem[d],
                                i32_val(multiDimCTAInRepId[d] * shapePerCTA[d] +
                                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentSizePerThread = getSizePerThread(parentEncoding);
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto offsets = emitOffsetForLayout(layout, type);
      auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
      SmallVector<int> idxs;
      for (SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
          parentEncoding, loc, rewriter, idxs[elemId], parentTy,
          sliceLayout.paddedShape(multiDimCTAInRepId),
          sliceLayout.paddedShape(shapePerCTA));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    if (auto mmaLayout = dyn_cast<MACAMmaEncodingAttr>(layout)) {
      SmallVector<Value> mmaColIdx(4);
      SmallVector<Value> mmaRowIdx(2);
      Value threadId = getThreadId(rewriter, loc);
      Value warpSize = i32_val(64);
      Value laneId = urem(threadId, warpSize);
      Value warpId = udiv(threadId, warpSize);
      auto elemsStride = mmaLayout.getElementsStride();
      auto isATrans = mmaLayout.getIsATrans();
      auto isBTrans = mmaLayout.getIsBTrans();
      // TODO: fix the bug in MMAEncodingAttr document

      SmallVector<Value> multiDimWarpId(2);
      multiDimWarpId[0] = urem(warpId, i32_val(mmaLayout.getWarpsPerCTA()[0]));
      multiDimWarpId[1] = udiv(warpId, i32_val(mmaLayout.getWarpsPerCTA()[0]));

      Value _1 = i32_val(1);
      Value _2 = i32_val(2);
      Value _4 = i32_val(4);
      Value _8 = i32_val(8);
      Value _16 = i32_val(16);

      assert(rank == 2);
      SmallVector<Value> multiDimOffset(rank);
      auto elemsMnk = mmaLayout.getElementsMNK();

      if (mmaLayout.getColMajor()) {
        multiDimWarpId[0] =
            urem(multiDimWarpId[0],
                 i32_val(ceil<unsigned>(shape[0], (16 * elemsMnk[0]))));
        multiDimWarpId[1] =
            urem(multiDimWarpId[1],
                 i32_val(ceil<unsigned>(shape[1], (16 * elemsMnk[1]))));
        Value rowWarpOffset =
            mul(multiDimWarpId[0], mul(_16, i32_val(elemsMnk[0])));
        Value colWarpOffset =
            mul(multiDimWarpId[1], mul(_16, i32_val(elemsMnk[1])));
        Value mmaRow;
        if (isATrans) {
          auto offset = mul(udiv(urem(laneId, _16), i32_val(elemsStride[0])),
                            i32_val(elemsStride[0] * elemsMnk[0]));
          mmaRow = add(add(urem(laneId, i32_val(elemsStride[0])),
                           mul(i32_val(elemId / (4 * elemsMnk[1])),
                               i32_val(elemsStride[0]))),
                       offset);
        } else {
          mmaRow = add(mul(urem(laneId, _16), i32_val(elemsMnk[0])),
                       i32_val(elemId / (4 * elemsMnk[1])));
        }
        Value mmaCol = i32_val(0);
        unsigned elemColId = elemId % (elemsMnk[1] * 4);

        if (byteWidth == 8) {
          mmaCol = udiv(laneId, _16);
        } else {
          Value mmaColGrp = udiv(laneId, _16);
          mmaCol = mul(mmaColGrp, mul(_4, i32_val(elemsMnk[1])));
        }

        mmaRowIdx[0] = add(mmaRow, rowWarpOffset);
        mmaColIdx[0] = add(mmaCol, colWarpOffset);

        if (byteWidth == 8) {
          multiDimOffset[1] = add(mmaColIdx[0], i32_val(elemColId * 4));
        } else {
          multiDimOffset[1] = add(mmaColIdx[0], i32_val(elemColId));
        }
        multiDimOffset[0] = mmaRowIdx[0];

        // add repplica offset
        multiDimOffset[0] = add(
            multiDimOffset[0], i32_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], i32_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      } else {
        multiDimWarpId[0] =
            urem(multiDimWarpId[0],
                 i32_val(ceil<unsigned>(shape[0], (16 * elemsMnk[0]))));
        multiDimWarpId[1] =
            urem(multiDimWarpId[1],
                 i32_val(ceil<unsigned>(shape[1], (16 * elemsMnk[1]))));
        Value rowWarpOffset =
            mul(multiDimWarpId[0], mul(_16, i32_val(elemsMnk[0])));
        Value colWarpOffset =
            mul(multiDimWarpId[1], mul(_16, i32_val(elemsMnk[1])));
        Value mmaCol;
        if (!isBTrans) {
          mmaCol = mul(urem(laneId, _16), i32_val(elemsMnk[1]));
        } else {
          unsigned elemColId = elemId % elemsMnk[1];
          auto offset = mul(udiv(urem(laneId, _16), i32_val(elemsStride[1])),
                            i32_val(elemsStride[1] * elemsMnk[1]));
          mmaCol = add(add(offset, i32_val(elemColId * elemsStride[1])),
                       urem(laneId, i32_val(elemsStride[1])));
        }
        Value mmaRow = i32_val(0);
        unsigned elemRowId = elemId / elemsMnk[1];

        if (byteWidth == 8) {
          mmaRow = udiv(laneId, _16);
        } else {
          Value mmaRowGrp = udiv(laneId, _16);
          mmaRow = mul(mmaRowGrp, mul(_4, i32_val(elemsMnk[0])));
        }

        mmaRowIdx[0] = add(mmaRow, rowWarpOffset);
        mmaColIdx[0] = add(mmaCol, colWarpOffset);

        if (byteWidth == 8) {
          multiDimOffset[0] = add(mmaRowIdx[0], i32_val(elemRowId * 4));
        } else {
          if (!isATrans) {
            multiDimOffset[0] = add(mmaRowIdx[0], i32_val(elemRowId));
          } else {
            auto elemOffset = (elemRowId % elemsMnk[0]) * elemsStride[0] +
                              (elemRowId / elemsMnk[0]) % elemsStride[0];
            multiDimOffset[0] = add(mmaRowIdx[0], i32_val(elemOffset));
          }
        }
        multiDimOffset[1] = mmaColIdx[0];

        // add repplica offset
        multiDimOffset[0] = add(
            multiDimOffset[0], i32_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], i32_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      }

      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase, triton::gpu::ConvertLayoutOp op,
                      ArrayRef<unsigned> elemsPerThread = {1, 1},
                      bool isMACALayout = true) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    SmallVector<unsigned> numCTAs(rank);
    auto shapePerCTA = getShapePerCTATile(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTAs[d] = ceil<unsigned>(type.getShape()[d], shapePerCTA[d]);
    }
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);

    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTAs, order);

      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        auto byteWidth = type.getElementType().isF64() ? 8 : 4;
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTA, byteWidth);
        auto shapePerCTATile = getShapePerCTATile(layout);
        auto origRepShape = getRepShapeForCvtLayout(op);
        auto getShapePerCta = getShapePerCTA(layout, type.getShape());
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            getShapePerCta);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
#ifndef USE_MACA_OPAQUE_PTR
        auto elemPtrTy = ptr_ty(rewriter.getContext(), llvmElemTy, 3);
#else
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
#ifndef USE_MACA_OPAQUE_PTR
        ptr = bitcast(ptr, ptr_ty(vecTy, 3));
#else
        ptr = bitcast(ptr, ptr_ty(rewriter.getContext(), 3));
#endif
        if (stNotRd) {
          Value valVec = undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1)
              currVal = zext(llvmElemTy, currVal);
            else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
          }
          store(valVec, ptr);
        } else {
          Value valVec = load(vecTy, ptr);

          for (unsigned v = 0; v < vec; ++v) {
            Value currVal = extract_element(llvmElemTy, valVec, i32_val(v));
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<LLVM::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
          }
        }
      }
    }
  }

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    //////////////////////////////////////////////////////////////////////////
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    MACAMmaEncodingAttr mmaLayout;

    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation(),
                                               getTypeConverter());
#ifndef USE_MACA_OPAQUE_PTR
    auto elemPtrTy = ptr_ty(rewriter.getContext(), llvmElemTy, 3);
#else
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
    smemBase = bitcast(smemBase, elemPtrTy);

    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    SmallVector<unsigned> elemsMnk(2);

    auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout, shape);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    auto srcMma = dyn_cast<MACAMmaEncodingAttr>(srcLayout);
    auto dstMma = dyn_cast<MACAMmaEncodingAttr>(dstLayout);
    if (srcMma && dstMma) {
      // TODO(MACA): why this? seems not support mma -> mma now
      if ((srcMma.getWarpsPerCTA() == dstMma.getWarpsPerCTA()) &&
          (srcMma.getElementsMNK() == dstMma.getElementsMNK()) &&
          (srcMma.getColMajor() == dstMma.getColMajor())) {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
    }

    bool isSrcMACA = false, isDstMACA = false;
    if (auto layout = dyn_cast<MACAMmaEncodingAttr>(srcLayout)) {
      mmaLayout = layout;
      isSrcMACA = true;
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(srcLayout)) {
      isSrcMACA = isa<MACAMmaEncodingAttr>(sliceLayout.getParent());
      if (isSrcMACA)
        mmaLayout = cast<MACAMmaEncodingAttr>(sliceLayout.getParent());
    }

    if (auto layout = dyn_cast<MACAMmaEncodingAttr>(dstLayout)) {
      mmaLayout = layout;
      isDstMACA = true;
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(dstLayout)) {
      isDstMACA = isa<MACAMmaEncodingAttr>(sliceLayout.getParent());
      if (isDstMACA)
        mmaLayout = cast<MACAMmaEncodingAttr>(sliceLayout.getParent());
    }

    if (mmaLayout) {
      elemsMnk[0] = mmaLayout.getElementsMNK()[0];
      elemsMnk[1] = mmaLayout.getElementsMNK()[1];
    }

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }

    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    unsigned inVec = 0;
    unsigned outVec = 0;
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    // TODO: output with column major to be supported.
    if (isSrcMACA && rank == 2 && shape[0] != shape[1] && outOrd[0] == 0 &&
        outOrd[1] == 1 && (elemsMnk[0] != 1 || elemsMnk[1] != 1))
      assert(false && "Mma output with column major to be supported!");

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0)
        barrier();
      if (isa<BlockedEncodingAttr>(srcLayout) ||
          isa<SliceEncodingAttr>(srcLayout) ||
          isa<MACAMmaEncodingAttr>(srcLayout)) {
        processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                       multiDimRepId, inVec, paddedRepShape, outOrd, vals,
                       smemBase, op, elemsMnk, isSrcMACA);
      } else {
        assert(0 && "ConvertLayout with input layout not implemented");
        return failure();
      }

      barrier();
      if (isa<BlockedEncodingAttr>(dstLayout) ||
          isa<SliceEncodingAttr>(dstLayout) ||
          isa<MACAMmaEncodingAttr>(dstLayout)) {
        processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                       outNumCTAsEachRep, multiDimRepId, outVec, paddedRepShape,
                       outOrd, outVals, smemBase, op, elemsMnk, isDstMACA);
      } else {
        assert(0 && "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  // Convert from accumulator MMA layout to 8bit dot operand layout.
  // The conversion logic is taken from:
  // https://github.com/ColfaxResearch/cutlass-kernels/blob/a9de6446c1c0415c926025cea284210c799b11f8/src/fmha-pipeline/reg2reg.h#L45
  void
  convertMMAV3To8BitsDotOperand(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    assert(0);
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
#ifdef USE_MACA
    if (isMmaToDotShortcut(srcTy, dstTy)) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
#else
    if (matchMmaV3AndDotOperandLayout(srcTy, dstTy)) {
      if (srcTy.getElementType().getIntOrFloatBitWidth() == 16) {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
      assert(srcTy.getElementType().getIntOrFloatBitWidth() == 8 &&
             "Unsupported type size.");
      convertMMAV3To8BitsDotOperand(op, adaptor, rewriter);
      return success();
    }

    if (isMmaToDotShortcut(srcTy, dstTy)) {
      // get source values
      auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      unsigned elems = getTotalElemsPerThread(srcTy);
      Type elemTy =
          this->getTypeConverter()->convertType(srcTy.getElementType());
      // for the destination type, we need to pack values together
      // so they can be consumed by tensor core operations
      SmallVector<Value> vecVals;
      SmallVector<Type> types;
      // For some reasons, LLVM's NVPTX backend inserts unnecessary (?) integer
      // instructions to pack & unpack sub-word integers. A workaround is to
      // store the results of ldmatrix in i32
      auto elemSize = elemTy.getIntOrFloatBitWidth();
      if (auto intTy = dyn_cast<IntegerType>(elemTy) && elemSize <= 16) {
        auto fold = 32 / elemSize;
        for (unsigned i = 0; i < elems; i += fold) {
          Value val = i32_val(0);
          for (unsigned j = 0; j < fold; j++) {
            auto ext =
                shl(i32_ty, zext(i32_ty, vals[i + j]), i32_val(elemSize * j));
            val = or_(i32_ty, val, ext);
          }
          vecVals.push_back(val);
        }
        elems = elems / (32 / elemSize);
        types = SmallVector<Type>(elems, i32_ty);
      } else {
        unsigned vecSize = std::max<unsigned>(32 / elemSize, 1);
        Type vecTy = vec_ty(elemTy, vecSize);
        types = SmallVector<Type>(elems / vecSize, vecTy);
        for (unsigned i = 0; i < elems; i += vecSize) {
          Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
          for (unsigned j = 0; j < vecSize; j++)
            packed = insert_element(vecTy, packed, vals[i + j], i32_val(j));
          vecVals.push_back(packed);
        }
      }

      // This needs to be ordered the same way that
      // ldmatrix.x4 would order it
      // TODO: this needs to be refactor so we don't
      // implicitly depends on how emitOffsetsForMMAV2
      // is implemented
      SmallVector<Value> reorderedVals;
      for (unsigned i = 0; i < vecVals.size(); i += 4) {
        reorderedVals.push_back(bitcast(vecVals[i], i32_ty));
        reorderedVals.push_back(bitcast(vecVals[i + 2], i32_ty));
        reorderedVals.push_back(bitcast(vecVals[i + 1], i32_ty));
        reorderedVals.push_back(bitcast(vecVals[i + 3], i32_ty));
      }

      Value view = packLLElements(loc, getTypeConverter(), reorderedVals,
                                  rewriter, dstTy);
      rewriter.replaceOp(op, view);
      return success();
    }
#endif
    return failure();
  }

private:
  const METAX::TargetInfo &targetInfo;
};

struct GVMArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GVMArriveOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::GVMArriveOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::GVMArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.arrive");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    num += (1 << 6);
    auto loc = op.getLoc();
    Value num_value = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, num);
    mlir::LLVM::createBuiltinFunc<triton::gpu::GVMArriveOp>(
        rewriter, loc, op, funcName, getVoidType(), {num_value});
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct SchedBoundOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::SchedBoundOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::SchedBoundOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::SchedBoundOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.schedbound.begin");
    auto loc = op.getLoc();
    Value voidVal = undef(void_ty(op.getContext()));
    // ValueRange voidVals = {voidVal};
    ValueRange voidVals = {};
    mlir::LLVM::createBuiltinFunc<triton::gpu::SchedBoundOp>(
        rewriter, loc, op, funcName, getVoidType(), voidVals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern<triton::gpu::BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.barrier.inst");
    auto loc = op.getLoc();
    Value voidVal = undef(void_ty(op.getContext()));
    // ValueRange voidVals = {voidVal};
    ValueRange voidVals = {};
    mlir::LLVM::createBuiltinFunc<triton::gpu::BarrierOp>(
        rewriter, loc, op, funcName, getVoidType(), voidVals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierSharedOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierSharedOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::BarrierSharedOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.barrier.shared");
    auto loc = op.getLoc();
    Value voidVal = undef(void_ty(op.getContext()));
    // ValueRange voidVals = {voidVal};
    ValueRange voidVals = {};
    mlir::LLVM::createBuiltinFunc<triton::gpu::BarrierSharedOp>(
        rewriter, loc, op, funcName, getVoidType(), voidVals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct IGLPOpConversion : public ConvertOpToLLVMPattern<triton::gpu::IGLPOp> {
  using ConvertOpToLLVMPattern<triton::gpu::IGLPOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::IGLPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef funcName("llvm.mxc.igroup.config");
    // config 0 : enable igroup optimization, default 0 (automatic by compiler).
    // config 1 : the number of prefetched lds instrs, default -1.
    // config 2 : the number of mma instrs between lds group, default -1.
    // config 3 : the number of mma instrs between sts group, default -1.
    // config 4 : the number of other instrs between mma instrs, default -1.
    // config 5 : the number of lds instrs inside lds group, default -1.
    // config 6 : the number of mma instrs between ldg instrs, default -1.
    // config 7 : the number of mma instrs between last lds and arrive, default
    // -1. [EXPERIMENTAL] when config 0 == 3:
    //   config 2 : the number of other instrs between lds group, default -1.
    //   config 3 : the number of other instrs between sts group, default -1.
    //   config 4 : the number of other instrs between mma instrs, default -1.
    //   config 5 : the number of other instrs between ldg instrs, default -1.
    auto config_0 = op->getAttrOfType<IntegerAttr>("config_0").getInt();
    auto config_1 = op->getAttrOfType<IntegerAttr>("config_1").getInt();
    auto config_2 = op->getAttrOfType<IntegerAttr>("config_2").getInt();
    auto config_3 = op->getAttrOfType<IntegerAttr>("config_3").getInt();
    auto config_4 = op->getAttrOfType<IntegerAttr>("config_4").getInt();
    auto config_5 = op->getAttrOfType<IntegerAttr>("config_5").getInt();
    auto config_6 = op->getAttrOfType<IntegerAttr>("config_6").getInt();
    auto config_7 = op->getAttrOfType<IntegerAttr>("config_7").getInt();
    auto loc = op.getLoc();
    Value val_0 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_0);
    Value val_1 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_1);
    Value val_2 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_2);
    Value val_3 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_3);
    Value val_4 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_4);
    Value val_5 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_5);
    Value val_6 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_6);
    Value val_7 = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, config_7);
    ValueRange vals = {val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7};
    mlir::LLVM::createBuiltinFunc<triton::gpu::IGLPOp>(
        rewriter, loc, op, funcName, getVoidType(), vals);
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::METAX::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit,
    IndexCacheInfoSm indexCacheInfoSm, bool enSmIdxCache, bool enSmIndexOpt) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, indexCacheInfoSm,
                                      enSmIdxCache, enSmIndexOpt, benefit);
}

void mlir::triton::METAX::populateSyncOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GVMArriveOpConversion>(typeConverter, benefit);
  patterns.add<SchedBoundOpConversion>(typeConverter, benefit);
  patterns.add<IGLPOpConversion>(typeConverter, benefit);
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
  patterns.add<BarrierSharedOpConversion>(typeConverter, benefit);
}
