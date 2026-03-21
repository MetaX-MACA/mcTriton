#include "TargetInfo.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace maca;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Return the mask for the unique data accessed by given tensor type.
// Used to mask out the redundant data accessed by threads.
Value redundantDataMask(Type valueTy, ConversionPatternRewriter &rewriter,
                        Location loc, const METAX::TargetInfo &targetInfo) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
  Value mask = int_val(1, 1);
  auto tid = tid_val();
  auto clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
  if (tensorTy) {
    auto layout = tensorTy.getEncoding();
    auto shape = tensorTy.getShape();
    unsigned rank = shape.size();
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
    auto order = triton::gpu::getOrder(layout);
    auto warpOrder = triton::gpu::getWarpOrder(layout);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
    Value warpSize = i32_val(64);
    Value laneId = urem(tid, warpSize);
    Value warpId = udiv(tid, warpSize);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    for (unsigned dim = 0; dim < rank; ++dim) {
      // if there is no data replication across threads on this dimension
      if (shape[dim] >= shapePerCTATile[dim])
        continue;
      // Otherwise, we need to mask threads that will replicate data on this
      // dimension. Calculate the thread index on this dimension for the CTA
      Value threadDim =
          add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
              multiDimThreadId[dim]);
      mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                 i32_val(shape[dim])));
    }
    // Do not write duplicated data when multicast is enabled
    if (triton::gpu::getNumCTAs(layout) > 1) {
      auto _0 = i32_val(0);
      auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
      auto CTASplitNum = triton::gpu::getCTASplitNum(layout);
      auto CTAOrder = triton::gpu::getCTAOrder(layout);

      auto multiDimClusterCTAId =
          delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

      for (unsigned dim = 0; dim < rank; ++dim) {
        // Skip when multicast is not enabled in this dimension
        if (CTAsPerCGA[dim] == CTASplitNum[dim])
          continue;
        // This wrapping rule must be consistent with emitCTAOffsetForLayout
        unsigned splitNum = std::min<unsigned>(shape[dim], CTASplitNum[dim]);
        Value repId = udiv(multiDimClusterCTAId[dim], i32_val(splitNum));
        // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
        //     CTA0 and CTA2 holds data of block0,
        //     CTA1 and CTA3 holds data of block1.
        // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
        // be masked. We add the following mask:
        //     multiDimClusterCTAId[dim] / splitNum == 0
        // Actually in all existing cases of multicast, splitNum is always 1.
        // The mask is equivalent to:
        //     multiDimClusterCTAId[dim] == 0
        mask = and_(mask, icmp_eq(repId, _0));
      }
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = and_(mask, icmp_eq(clusterCTAId, i32_val(0)));
    mask = and_(mask, icmp_eq(tid, i32_val(0)));
  }
  return mask;
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const METAX::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getThreadConstRepeatTimes(Value ptr) const {
    if (getenv("TRITON_DISABLE_CONSTANCY_LOAD_LAYOUT_OPT")) {
      return 1;
    }
    // constancy > 1 and sizePerThread > 1 and (sizePerThread % constancy == 0)
    // can return value > 1
    auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto layout = tensorTy.getEncoding();
    auto order = triton::gpu::getOrder(layout);
    auto sizePerThread = triton::gpu::getSizePerThread(layout)[order[0]];
    auto constancy = axisInfo->getConstancy(order[0]);
    // constancy > sizePerThread, return sizePerThread
    if ((constancy > sizePerThread) && (constancy % sizePerThread == 0))
      return sizePerThread;
    if (sizePerThread % constancy)
      return 1;
    return constancy;
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // The maximum vector size is 128 bits on METAX GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned
  getVectorSize(Value ptr,
                llvm::ArrayRef<int64_t> contiguityInterConstGroup) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto constRepeatPerThread = getThreadConstRepeatTimes(ptr);

    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    if (contiguity == 1 && constRepeatPerThread > 1) {
      auto layout = tensorTy.getEncoding();
      auto order = triton::gpu::getOrder(layout);
      auto sizePerThread = triton::gpu::getSizePerThread(layout)[order[0]];
      auto loadPerThread =
          std::max<unsigned>(sizePerThread / constRepeatPerThread, 1);
      contiguity = std::min<unsigned>(contiguityInterConstGroup[order[0]],
                                      loadPerThread);
    }
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);

    // The maximum vector size is 128 bits on METAX GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

  unsigned getAxisInfoContiguity(Value ptr, int dim) const {
    auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
    return axisInfo->getContiguity(dim);
  }

protected:
  const METAX::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

namespace maca {
void appendIntrinsicModifer(std::string &str, int vec, Type elemType) {
  str += ".";
  if (vec > 1) {
    str += "v";
    str += std::to_string(vec);
  }
  if (elemType.isF32()) {
    str += "f32";
  } else if (elemType.isF64()) {
    str += "f64";
  } else if (elemType.isF16()) {
    str += "f16";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 64) {
    str += "i64";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 32) {
    str += "i32";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 16) {
    str += "i16";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 8) {
    str += "i8";
  } else if (elemType.isFloat8E5M2() || elemType.isFloat8E4M3FN()) {
    str += "i8";
  } else {
    assert(false && "Intrinsic Load unsupported data type");
  }
}
} // namespace maca

namespace {

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const METAX::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto typeConverter = getTypeConverter();

    // original valuesgetVectorSize
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    // TODO: Get contiguityInterConstGroup from LoadOp
    auto contiguityPerConstArray = op.getContiguityInterConstGroup();
    unsigned vec;
    if (contiguityPerConstArray.size() > 0) {
      vec = getVectorSize(ptr, contiguityPerConstArray);
    } else {
      vec = getVectorSize(ptr);
    }

    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    unsigned constRepeatPerThread = getThreadConstRepeatTimes(ptr);

    if (llMask) {
      LLVM_DEBUG(DBGS() << "vec = " << vec
                        << " mask_alignment = " << getMaskAlignment(mask));
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      constRepeatPerThread = std::min<unsigned>(constRepeatPerThread, getMaskAlignment(mask));
      LLVM_DEBUG(llvm::dbgs() << " vec = " << vec << '\n');
    }

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConst = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConst = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    float splatValfp = 0;
    if (other &&
        (isa<FloatType>(valueElemTy) || isa<IntegerType>(valueElemTy)) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<FloatType>(constAttr.getElementType())) {
      otherIsSplatConst = true;
      if (isa<Float64Type>(valueElemTy)) {
        splatValfp = static_cast<float>(
            constAttr.getSplatValue<APFloat>().convertToDouble());
      } else {
        splatValfp = constAttr.getSplatValue<APFloat>().convertToFloat();
      }
    }
    bool isOtherValid = other ? false : true;
    if (otherIsSplatConst && (splatVal == 0) && (splatValfp == 0))
      isOtherValid = true;

    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    LDBG("LoadOp numElems = " << numElems << " vec = " << vec
                              << " valueElemNBits = " << valueElemNBits << " "
                              << op.getType());
    SmallVector<Value> loadedVals;

    const char *disableOptFlag = getenv("TRITON_DISABLE_LOAD_STORE_OPT");
    const char *disableLdgPred = getenv("TRITON_DISABLE_LDG_PREDICATOR");

    int stride;
    if (constRepeatPerThread > 1) {
      stride = vec * constRepeatPerThread;
    } else {
      stride = vec;
    }

    for (size_t vecStart = 0; vecStart < numElems; vecStart += stride) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits); // 32
      const size_t totalWidth = valueElemNBits * vec;          // 16*8 = 128
      const size_t width = std::min(totalWidth, maxWordWidth); // 32
      const size_t nWords = std::max<size_t>(1, totalWidth / width); // 4
      const size_t wordNElems = width / valueElemNBits;              // 2
      assert(wordNElems * nWords * numVecs == numElems);
      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);
      Value zeroVal = bitcast(int_val(valueElemNBits, 0), valueElemTy);
      bool useIntrinsic = op.getIntrinsic();
      // todo: support llvm.mxc.load.global.async
      if (!disableOptFlag) {
        if (!disableLdgPred && isOtherValid &&
            (totalWidth == 128 || totalWidth == 64 || totalWidth == 32 ||
             totalWidth == 16 || totalWidth == 8)) {
          // llvm.mxc.ldg.predicator
          Type retTy =
              vec > 1 ? vec_ty(valueElemTy, vec)
                      : valueElemTy; // RetType of llvm.mxc.ldg.predicator.f32
                                     // should be float
#ifndef USE_MACA_OPAQUE_PTR
          Type retPtrTy = ptr_ty(retTy, 1);
#else
          Type retPtrTy = ptr_ty(rewriter.getContext(), 1);
#endif
          // "llvm.mxc.icmp.i64.i32" builtin function calculate ldg mask
          // ldg mask must be int64 streg
          // convert i1 maskElems[vecStart] to i32
          Value maskElem =
              mask ? zext(IntegerType::get(rewriter.getContext(), 32),
                          maskElems[vecStart])
                   : int_val(32, 1);
          std::string icmp = "llvm.mxc.icmp.i64.i32";
          StringRef icmpName(icmp);
          Value cmpModel = i32_val(34);
          SmallVector<Value> rangeValueCmp(3);
          rangeValueCmp[0] = maskElem;       // mask to be compared
          rangeValueCmp[1] = int_val(32, 0); // compare to 0
          rangeValueCmp[2] = cmpModel;       // compare model
          ValueRange icomValue(rangeValueCmp);

          Value xmask = mlir::LLVM::createBuiltinFunc<triton::LoadOp>(
              rewriter, loc, op, icmpName, i64_ty, icomValue);

          std::string ldgPredicator = "llvm.mxc.ldg.predicator";
          appendIntrinsicModifer(ldgPredicator, vec, valueElemTy);
          StringRef predictName(ldgPredicator);
          SmallVector<Value> rangeValueLdg(7);
          Value ldgPtr = bitcast(ptrElems[vecStart], retPtrTy);
          rangeValueLdg[0] = ldgPtr;     // global addr, default mtreg
          rangeValueLdg[1] = i32_val(0); // addr offset, immediate number

          rangeValueLdg[2] = xmask;         // mask
          rangeValueLdg[3] = int_val(1, 1); // enable return0
          if (std::getenv("TRITON_DISABLE_LDG_SADDR") != nullptr) {
            rangeValueLdg[4] = int_val(1, 0); // use saddr flag
          } else {
            rangeValueLdg[4] = int_val(1, 1); // use saddr flag
          }
          // rangeValueLdg[4] = int_val(1, 0); // use saddr flag
          rangeValueLdg[5] = int_val(1, 0); // pred_neg flag
          rangeValueLdg[6] =
              useIntrinsic ? int_val(1, 1) : int_val(1, 0); // enable async

          ValueRange ldgValue(rangeValueLdg);

          Value ldg_values = mlir::LLVM::createBuiltinFunc<triton::LoadOp>(
              rewriter, loc, op, predictName, retTy, ldgValue);
          if (vec == 1) {
            for (int i = 0; i < constRepeatPerThread; i++) {
              loadedVals.push_back(ldg_values);
            }
          } else {
            for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
              Value curr = extract_element(ldg_values, i32_val(elemIndex));
              for (int i = 0; i < constRepeatPerThread; i++) {
                loadedVals.push_back(curr);
              }
            }
          }
        } else { // llvm.mxc.load.global.async
          Type retTy = vec > 1 ? vec_ty(valueElemTy, vec) : valueElemTy;
#ifndef USE_MACA_OPAQUE_PTR
          Type retPtrTy = ptr_ty(retTy, 1);
#else
          Type retPtrTy = ptr_ty(rewriter.getContext(), 1);
#endif
          auto loaded = rewriter.create<scf::IfOp>(
              loc, pred,
              [&](OpBuilder &builder, Location loc) {
                Value vec_values;
                if (useIntrinsic) {
                  std::string intrinsicName = "llvm.mxc.load.global.async";
                  appendIntrinsicModifer(intrinsicName, vec, valueElemTy);
                  StringRef funcName(intrinsicName);
                  Value tt = bitcast(ptrElems[vecStart], retPtrTy);
                  vec_values = mlir::LLVM::createBuiltinFunc<triton::LoadOp>(
                      rewriter, loc, op, funcName, retTy, {tt});
                } else {
                  vec_values =
                      load(retTy, bitcast(ptrElems[vecStart], retPtrTy));
                }
                builder.create<mlir::scf::YieldOp>(loc,
                                                   ValueRange({vec_values}));
              },
              [&](OpBuilder &builder, Location loc) {
                Value vec_values = undef(retTy);
                if (vec == 1) {
                  Value otherVal = other ? otherElems[vecStart] : zeroVal;
                  vec_values = otherVal;
                } else {
                  for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
                    size_t elemOffset = vecStart + elemIndex;
                    Value otherVal = other ? otherElems[elemOffset] : zeroVal;
                    vec_values = insert_element(retTy, vec_values, otherVal,
                                                i32_val(elemIndex));
                  }
                }
                builder.create<mlir::scf::YieldOp>(loc,
                                                   ValueRange({vec_values}));
              });
          Value ret = loaded->getResult(0);
          if (vec == 1) {
            for (int i = 0; i < constRepeatPerThread; i++) {
              loadedVals.push_back(ret);
            }
          } else {
            for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
              Value curr = extract_element(ret, i32_val(elemIndex));
              for (int i = 0; i < constRepeatPerThread; i++) {
                loadedVals.push_back(curr);
              }
            }
          }
        }
      } else {
        for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
          for (size_t wordElem = 0; wordElem < wordNElems; ++wordElem) {
            size_t elemOffset = vecStart + wordIdx * wordNElems + wordElem;
            auto loaded = rewriter.create<scf::IfOp>(
                loc, pred,
                [&](OpBuilder &builder, Location loc) {
                  auto loadVal = builder.create<LLVM::LoadOp>(
                      loc, valueElemTy, ptrElems[elemOffset]);
                  builder.create<scf::YieldOp>(loc, ValueRange({loadVal}));
                },
                [&](OpBuilder &builder, Location loc) {
                  Value otherVal = other ? otherElems[elemOffset] : zeroVal;
                  builder.create<scf::YieldOp>(loc, ValueRange({otherVal}));
                });
            loadedVals.push_back(loaded->getResult(0));
          }
        }
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const METAX::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // C500-30592 bug fix: typeConverter->convertType implictly convert bf16 ptr
    // and bf16 value to int16 ptr and int16 value, but the value of constant is
    // not converted, to fix it we convert the value of constant explicitly to
    // make them .
    if (valueElemTy.isInteger(16)) {
      for (size_t elemIndex = 0; elemIndex < elemsPerThread; elemIndex++) {
        Value elem = valueElems[elemIndex];
        // check if value is constant and bf16
        if (auto constantOp =
                dyn_cast_or_null<LLVM::ConstantOp>(elem.getDefiningOp())) {
          auto valueAttr = constantOp.getValue();
          auto floatAttr = dyn_cast_or_null<mlir::FloatAttr>(valueAttr);
          if (floatAttr && floatAttr.getType().isBF16()) {
            // create new int16 constant value and replace the origin bf16 value
            llvm::APFloat bf16Value = floatAttr.getValue();
            llvm::APInt apInt = bf16Value.bitcastToAPInt();
            int16_t int16Value = static_cast<int16_t>(apInt.getZExtValue());
            mlir::Type int16Type = rewriter.getIntegerType(16);
            mlir::Attribute newValueAttr =
                rewriter.getI16IntegerAttr(int16Value);
            auto newConstantOp = rewriter.create<LLVM::ConstantOp>(
                constantOp.getLoc(), int16Type, newValueAttr);
            rewriter.replaceOp(constantOp, newConstantOp.getResult());
          }
        }
      }
    }

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    const int numVecs = elemsPerThread / vec;
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);
      const char *disableOptFlag = getenv("TRITON_DISABLE_LOAD_STORE_OPT");
      const char *disableStgPred = getenv("TRITON_DISABLE_STG_PREDICATOR");
      if (!disableOptFlag) {
        if (!disableStgPred &&
            (totalWidth == 128 || totalWidth == 64 || totalWidth == 32)) {
          Type retTy = vec > 1 ? vec_ty(valueElemTy, vec) : valueElemTy;
#ifndef USE_MACA_OPAQUE_PTR
          Type retPtrTy = ptr_ty(IntegerType::get(rewriter.getContext(), 8), 1);
#else
          Type retPtrTy = ptr_ty(rewriter.getContext(), 1);
#endif
          // global addr
          Value stgPtr = bitcast(ptrElems[vecStart], retPtrTy);
          // data to be stored
          Value vec_values = undef(retTy);
          if (vec > 1) {
            for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
              Value elem = valueElems[vecStart + elemIndex];
              vec_values =
                  insert_element(retTy, vec_values, elem, i32_val(elemIndex));
            }
          } else {
            vec_values = valueElems[vecStart];
          }
          // mask
          Value maskVal = llMask ? and_(mask, maskElems[vecStart]) : mask;
          Value maskElem =
              zext(IntegerType::get(rewriter.getContext(), 32), maskVal);
          std::string icmp = "llvm.mxc.icmp.i64.i32";
          StringRef icmpName(icmp);
          Value cmpModel = i32_val(34);
          SmallVector<Value> rangeValueCmp(3);
          rangeValueCmp[0] = maskElem;       // mask to be compared
          rangeValueCmp[1] = int_val(32, 0); // compare to 0
          rangeValueCmp[2] = cmpModel;       // compare model
          ValueRange icomValue(rangeValueCmp);

          Value xmask = mlir::LLVM::createBuiltinFunc<triton::StoreOp>(
              rewriter, loc, op, icmpName, i64_ty, icomValue);

          std::string stgPredicator = "llvm.mxc.stg.predicator";
          appendIntrinsicModifer(stgPredicator, vec, valueElemTy);
          StringRef stgPredictName(stgPredicator);

          SmallVector<Value> rangeValueStg(7);
          rangeValueStg[0] = stgPtr;     // global addr
          rangeValueStg[1] = i32_val(0); // global addr offset, immediate number
          rangeValueStg[2] = vec_values; // data
          rangeValueStg[3] = xmask;      // mask
          if (std::getenv("TRITON_DISABLE_STG_SADDR") != nullptr) {
            rangeValueStg[4] = int_val(1, 0); // use saddr flag
          } else {
            rangeValueStg[4] = int_val(1, 1); // use saddr flag
          }
          // rangeValueStg[4] = int_val(1, 0); // use saddr flag
          rangeValueStg[5] = int_val(1, 0); // pred_neg flag
          rangeValueStg[6] = int_val(1, 0); // enable async
          ValueRange sdgValue(rangeValueStg);

          mlir::LLVM::createBuiltinFunc<triton::StoreOp>(
              rewriter, loc, op, stgPredictName, getVoidType(), sdgValue);
        } else {
          Value maskVal = llMask ? and_(mask, maskElems[vecStart]) : mask;
          Type retTy = vec_ty(valueElemTy, vec);
#ifndef USE_MACA_OPAQUE_PTR
          Type retPtrTy = ptr_ty(retTy, 1);
#else
          Type retPtrTy = ptr_ty(rewriter.getContext(), 1);
#endif
          rewriter.create<scf::IfOp>(
              loc, maskVal,
              [&](OpBuilder &builder, Location loc) {
                Value vec_values = undef(retTy);
                for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
                  Value elem = valueElems[vecStart + elemIndex];
                  vec_values = insert_element(retTy, vec_values, elem,
                                              i32_val(elemIndex));
                }
                store(vec_values, bitcast(ptrElems[vecStart], retPtrTy));
                builder.create<mlir::scf::YieldOp>(loc);
              },
              nullptr);
        }
      } else {
        for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
          // llWord is a width-len composition
          Value llWord = undef(wordTy);
          // Insert each value element to the composition
          for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
            const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
            assert(elemOffset < valueElems.size());
            Value elem = valueElems[elemOffset];
            if (elem.getType().isInteger(1))
              elem = rewriter.create<LLVM::SExtOp>(loc, type::i8Ty(ctx), elem);
            elem = bitcast(elem, valueElemTy);
            Value maskVal = llMask ? and_(mask, maskElems[vecStart]) : mask;
            // rewriter.create<scf::IfOp>(loc, std::nullopt, maskVal,
            rewriter.create<scf::IfOp>(
                loc, maskVal,
                [&](OpBuilder &builder, Location loc) {
                  auto storeOp = builder.create<LLVM::StoreOp>(
                      loc, elem, ptrElems[elemOffset]);
                  builder.create<scf::YieldOp>(loc);
                },
                nullptr);
          }
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

static LLVM::AtomicOrdering getMemoryOrdering(MemSemantic memOrdering) {
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return LLVM::AtomicOrdering::acq_rel;
  }
}

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::AtomicCASOp>::ConvertOpToLLVMPattern;

  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const METAX::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // extract relevant info from Module
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto TensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        TensorTy ? getTypeConverter()->convertType(TensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
    // tensor
    if (TensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];

      // as cmpx only support int type, here convert to int if it is not
      // fix "'llvm.cmpxchg' op operand #1 must be int" error
      Type intType;
      if (triton::type::isFloat(valueElemTy)) {
        if (valueElemTy.isF64()) {
          intType = rewriter.getI64Type(); // f64 → i64
        } else if (valueElemTy.isF32()) {
          intType = rewriter.getI32Type(); // f32 → i32
        } else if (valueElemTy.isF16() || valueElemTy.isBF16()) {
          intType = rewriter.getI16Type(); // f16/bf16 → i16
        } else {
          return failure();
        }

        casCmp = rewriter.create<LLVM::BitcastOp>(op.getLoc(), intType, casCmp);
        casVal = rewriter.create<LLVM::BitcastOp>(op.getLoc(), intType, casVal);
#ifndef USE_MACA_OPAQUE_PTR
        Type intPtrType = LLVM::LLVMPointerType::get(
            intType,
            cast<LLVM::LLVMPointerType>(casPtr.getType()).getAddressSpace());
#else
        Type intPtrType = LLVM::LLVMPointerType::get(
            rewriter.getContext(),
            cast<LLVM::LLVMPointerType>(casPtr.getType()).getAddressSpace());
#endif
        casPtr =
            rewriter.create<LLVM::BitcastOp>(op.getLoc(), intPtrType, casPtr);
      }

      // use op
      if (TensorTy) { // for tensor
        auto retType = vec == 1 ? valueElemTy : vecTy;
        // TODO: USE ATOMIC CAS OP on Tensor
        auto successOrdering = atomicMemOrdering;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef("device"));

        // Extract the new_loaded value from the pair.
        Value ret = nullptr;
        if (triton::type::isFloat(valueElemTy)) {
          ret = extract_val(intType, cmpxchg, i);
          ret = rewriter.create<LLVM::BitcastOp>(op.getLoc(), valueElemTy, ret);
        } else {
          ret = extract_val(valueElemTy, cmpxchg, i);
        }

        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
        }
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation(),
                                            getTypeConverter());
#ifndef USE_MACA_OPAQUE_PTR
        atomPtr =
            bitcast(atomPtr, ptr_ty(rewriter.getContext(), valueElemTy, 3));
#else
        atomPtr = bitcast(atomPtr, ptr_ty(rewriter.getContext(), 3));
#endif
        auto tid = tid_val();
        Value pred = icmp_eq(tid, i32_val(i));
        rewriter.create<LLVM::CondBrOp>(loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef("device"));
        // Extract the new_loaded value from the pair.
        Value newLoaded = nullptr;
        if (triton::type::isFloat(valueElemTy)) {
          newLoaded = extract_val(intType, cmpxchg, 0);
          newLoaded = rewriter.create<LLVM::BitcastOp>(op.getLoc(), valueElemTy,
                                                       newLoaded);
        } else {
          newLoaded = extract_val(valueElemTy, cmpxchg, 0);
        }

        store(newLoaded, atomPtr);

        rewriter.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        // add arrive(0);
        StringRef funcName("llvm.mxc.arrive");
        auto loc = op.getLoc();
        Value num_value = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, 0);
        mlir::LLVM::createBuiltinFunc<triton::AtomicCASOp>(
            rewriter, loc, op, funcName, getVoidType(), {num_value});

        barrier();
        Value ret = load(valueElemTy, atomPtr);
        barrier();
        rewriter.replaceOp(op, {ret});
      }
    }

    // replace op
    if (TensorTy) {
      Type structTy = getTypeConverter()->convertType(TensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::AtomicRMWOp>::ConvertOpToLLVMPattern;

  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const METAX::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  /// Try to match the mlir::triton::RMWOp to LLVM::AtomicBinOp.
  static std::optional<LLVM::AtomicBinOp> matchAtomicOp(RMWOp atomicOp) {
    switch (atomicOp) {
    case RMWOp::AND:
      return LLVM::AtomicBinOp::_and;
    case RMWOp::OR:
      return LLVM::AtomicBinOp::_or;
    case RMWOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case RMWOp::ADD:
      return LLVM::AtomicBinOp::add;
    case RMWOp::FADD:
      return LLVM::AtomicBinOp::fadd;
    case RMWOp::MAX:
      return LLVM::AtomicBinOp::max;
    case RMWOp::MIN:
      return LLVM::AtomicBinOp::min;
    case RMWOp::UMAX:
      return LLVM::AtomicBinOp::umax;
    case RMWOp::UMIN:
      return LLVM::AtomicBinOp::umin;
    case RMWOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    default:
      return std::nullopt;
    }
    llvm_unreachable("Invalid RMWOp");
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.getAtomicRmwOp();
    Value ptr = op.getPtr();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    Value opResult = op.getResult();
    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();
    const size_t valueElemNbits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    mask = and_(mask,
                icmp_slt(mul(tid, i32_val(elemsPerThread)), i32_val(numElems)));

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    auto vecTy = vec_ty(valueElemTy, vec);
    auto retType = vec == 1 ? valueElemTy : vecTy;
    SmallVector<Value> resultVals(elemsPerThread);
    const bool f16v2 = vec == 2 && valueElemTy.isF16();
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? and_(mask, maskElements[i]) : mask;

      Value undefVal = undef(retType);
      // Build blocks to bypass the atomic instruction for ~rmwMask.
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument({retType}, {loc});

      rewriter.setInsertionPointToEnd(curBlock);
      rewriter.create<LLVM::CondBrOp>(loc, rmwMask, atomicBlock, endBlock,
                                      undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      auto maybeKind = matchAtomicOp(atomicRmwAttr);
      Value atom;
      if (atomicRmwAttr == RMWOp::FADD && tensorTy &&
          memOrdering == MemSemantic::ACQUIRE_RELEASE &&
          tensorTy.getElementType().isBF16()) {
#ifdef USE_LLVM19_INTRINSIC
        std::string intrinsicName = "llvm.mxc.gvm.atomic.add.bf16.i16";
#else
        std::string intrinsicName = "llvm.mxc.gvm.atomic.add.bf16";
#endif
        Type retTy = valueElemTy;
        for (size_t j = 0; j < vec; ++j) {
          SmallVector<Value> rangeValue(2);
          rangeValue[0] = rmwPtr;
          rangeValue[1] = valElements[i + j];
          ValueRange atomicValue(rangeValue);
          Value atomic_val = mlir::LLVM::createBuiltinFunc<triton::AtomicRMWOp>(
              rewriter, loc, op, intrinsicName, retTy, atomicValue);
          atom = vec == 1 ? atomic_val
                          : insert_element(vecTy, undef(vecTy), atomic_val,
                                           i32_val(j));
        }
      } else {
        // TODO: use rocdl.raw.buffer.atomic from ROCDL dialect to use efficient
        // atomics for MI-* series of METAX GPU.
        atom = rewriter
                   .create<LLVM::AtomicRMWOp>(loc, *maybeKind, rmwPtr,
                                              valElements[i], atomicMemOrdering,
                                              StringRef("device"))
                   .getResult();

        // NV for the f16v2 case generates one packed instruction. We have to
        // create two separate instructions since LLVM::AtomicRMWOp doesn't
        // support this. Can be optimized out with rocdl.raw.buffer.atomic.
        if (f16v2) {
          Value atom2 =
              rewriter
                  .create<LLVM::AtomicRMWOp>(
                      loc, *maybeKind, ptrElements[i + 1], valElements[i + 1],
                      atomicMemOrdering, StringRef("device"))
                  .getResult();
          auto tmp = insert_element(vecTy, undef(vecTy), atom, i32_val(0));
          atom = insert_element(vecTy, tmp, atom2, i32_val(1)).getResult();
        }
      }
      if (!tensorTy) {
        Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation(),
                                            getTypeConverter());
// TODO(MACA): seems not right when vec != 1
#ifndef USE_MACA_OPAQUE_PTR
        Type atomPtrTy = ptr_ty(valueElemTy, 3);
#else
        Type atomPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
        store(atom, bitcast(atomPtr, atomPtrTy));
      }
      rewriter.create<LLVM::BrOp>(loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      Value retVal = endBlock->getArgument(0);
      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? retVal
                       : extract_element(valueElemTy, retVal, i32_val(ii));
        }
      } else {
        Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation(),
                                            getTypeConverter());
        barrier();
#ifndef USE_MACA_OPAQUE_PTR
        Type atomPtrTy = ptr_ty(valueElemTy, 3);
#else
        Type atomPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
        Value ret = load(valueElemTy, bitcast(atomPtr, atomPtrTy));
        barrier();
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const METAX::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    Value src = op.getSrc();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<MemDescType>(op.getResult().getType());
    auto srcElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto srcLayout = srcTy.getEncoding();
    assert((isa<BlockedEncodingAttr, SliceEncodingAttr>(srcLayout) &&
            "Unexpected srcLayout in AsyncCopyGlobalToLocalOpConversion"));
    auto srcBlockedLayout = cast<BlockedEncodingAttr>(srcTy.getEncoding());
    auto resSharedLayout = cast<SharedEncodingAttr>(dstTy.getEncoding());
    auto srcShape = srcTy.getShape();
    assert((srcShape.size() <= 3) &&
           "insert_slice_async: Unexpected rank of %src");

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // %src
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    // %dst
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      // FIXME(Keren): always assume other is 0 for now
      // It's not necessary for now because the pipeline pass will skip
      // generating insert_slice_async if the load op has any "other" tensor.
      // assert(false && "insert_slice_async: Other value not supported yet");
      otherElems = unpackLLElements(loc, llOther, rewriter);
      assert(srcElems.size() == otherElems.size());
    }

    // We don't use getVec() here because we are copying from memory to memory.
    // If contiguity > vector size, we can have one pointer maintaining the
    // start of the vector and the other pointer moving to the next vector.
    unsigned inVec = getContiguity(op.getSrc());
    auto inOrder = srcBlockedLayout.getOrder();
    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    if (maxPhase == 1) {
      minVec = inVec;
    }

    unsigned numElems = getTotalElemsPerThread(srcTy);
    auto srcEncoding = srcTy.getEncoding();
    SmallVector<Value> offsetVals = {smemObj.strides.size(), i32_val(0)};
    inOrder = triton::gpu::getOrder(srcEncoding);

    auto sizePerThread = triton::gpu::getSizePerThread(srcBlockedLayout);
    int numElemsPerThreadContigue = 0;
    for (int i = 0; i < inOrder.size(); i++) {
      if (inOrder[i] == 0) {
        numElemsPerThreadContigue = sizePerThread[i];
        break;
      }
    }

    // Determine if there is a mask.
    // Actually, after pipeline optimization, if there is a mask,
    // the 'mask' will be converted into an 'andIOp' or 'ExtractTensorOp',
    // so determine whether there is a 'mask' by the following method.
    bool isContainMask = false;
    if (isa<arith::AndIOp>(mask.getDefiningOp())) {
      isContainMask = true;
    }

    if (auto extractTensorOp =
            dyn_cast<triton::gpu::ExtractTensorOp>(mask.getDefiningOp())) {
      if (isa<arith::AndIOp>(extractTensorOp.getSource().getDefiningOp())) {
        isContainMask = true;
      }
    }

    // 16 * 8 = 128bits
    auto maxBitWidth =
        std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
    auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
    auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
    auto byteWidth = bitWidth / 8;
    assert(byteWidth == 16 || byteWidth == 8 || byteWidth == 4 ||
           byteWidth == 2 || byteWidth == 1);

    DenseMap<unsigned, Value> sharedNoSwiPtrs = getNoSwizzledSharedPtrs(
        loc, rewriter, targetInfo, op, inVec, srcTy, resSharedLayout, resElemTy,
        smemObj, offsetVals, smemObj.strides);
    // bool recalMaskSucc = true;
    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      auto srcPtrSwiBase = getContiguityGvmStartAddr(
          loc, rewriter, srcElems[elemIdx], srcTy, dstTy, typeConverter);

      auto srcSwiAddr = getCopyAsyncSwizzledGvmPtrs(
          loc, rewriter, targetInfo, inVec, elemIdx, srcTy, resSharedLayout,
          srcElemTy, srcPtrSwiBase);

#ifndef USE_MACA_OPAQUE_PTR
      Type bsmi8Ptr = ptr_ty(i8_ty, 3);
#else
      Type bsmi8Ptr = ptr_ty(rewriter.getContext(), 3);
#endif
      Value basePtr = sharedNoSwiPtrs[elemIdx];
      Value bsmbaseptr = bitcast(basePtr, bsmi8Ptr);

#ifndef USE_MACA_OPAQUE_PTR
      Type gvmi8Ptr = ptr_ty(i8_ty, 1);
#else
      Type gvmi8Ptr = ptr_ty(rewriter.getContext(), 1);
#endif
      Value gvmbaseptr = bitcast(srcSwiAddr, gvmi8Ptr);

      // ldg mask must be int64 streg
      // convert i1 maskElems[vecStart] to i32
      Value maskElem =
          zext(IntegerType::get(rewriter.getContext(), 32), maskElems[elemIdx]);
      std::string icmp = "llvm.mxc.icmp.i64.i32";
      StringRef icmpName(icmp);
      Value cmpModel = i32_val(34);
      SmallVector<Value> rangeValueCmp(3);
      rangeValueCmp[0] = maskElem;       // mask to be compared
      rangeValueCmp[1] = int_val(32, 0); // compare to 0
      rangeValueCmp[2] = cmpModel;       // compare model
      ValueRange icomValue(rangeValueCmp);
      Value xmask =
          mlir::LLVM::createBuiltinFunc<triton::gpu::AsyncCopyGlobalToLocalOp>(
              rewriter, loc, op, icmpName, i64_ty, icomValue);

      SmallVector<Value> rangeValueLdgBsm(8);
      rangeValueLdgBsm[0] = bsmbaseptr;    // shared memory addr
      rangeValueLdgBsm[1] = gvmbaseptr;    // gvmbaseptr;
      rangeValueLdgBsm[2] = i32_val(0);    // addr offset, default 0
      rangeValueLdgBsm[3] = xmask;         // mask
      rangeValueLdgBsm[4] = int_val(1, 1); // enable return0
      rangeValueLdgBsm[5] = int_val(1, 1); // flag: whether to use saddr
      rangeValueLdgBsm[6] = int_val(1, 0); // pred_neg
      rangeValueLdgBsm[7] = int_val(1, 1); // enable async
      ValueRange ldgBsmValue(rangeValueLdgBsm);

      std::string ldgPredicator = "llvm.mxc.ldg.predicator.bsm";
      Type valueTy = res.getType();
      Type valueElemTy =
          typeConverter->convertType(getElementTypeOrSelf(valueTy));
      appendIntrinsicModifer(ldgPredicator, minVec, valueElemTy);
      StringRef ldgBsmFuncName(ldgPredicator);

      // RetType of llvm.mxc.ldg.predicator.f32 should be float
      Type retTy = minVec > 1 ? vec_ty(valueElemTy, minVec) : valueElemTy;
      mlir::LLVM::createBuiltinFunc<triton::gpu::AsyncCopyGlobalToLocalOp>(
          rewriter, loc, op, ldgBsmFuncName, retTy, ldgBsmValue);
    }

    rewriter.replaceOp(op, llDst);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // AsyncWait = GVMArrive + BarrierShared
    StringRef funcName("llvm.mxc.arrive");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    num += (1 << 6);
    auto loc = op.getLoc();
    Value num_value = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, num);
    mlir::LLVM::createBuiltinFunc<triton::gpu::AsyncWaitOp>(
        rewriter, loc, op, funcName, getVoidType(), {num_value});
    return success();
  }
};

} // namespace

void mlir::triton::METAX::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion>(typeConverter, targetInfo,
                                                    axisInfoAnalysis, benefit);
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
  patterns.add<AsyncCopyGlobalToLocalOpConversion>(typeConverter, targetInfo,
                                                   axisInfoAnalysis, benefit);
  //   patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);
}
