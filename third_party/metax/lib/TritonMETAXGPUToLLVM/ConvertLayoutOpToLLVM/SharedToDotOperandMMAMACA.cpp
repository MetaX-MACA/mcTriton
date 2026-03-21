/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 */
#include "../MACACommonConversion.h"
#include "../Utility.h"
#include "./PatternTritonGPUOpToLLVM.h"
#include "ExtractStrideAndOffsetIntValue.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace maca;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using MmaEncodingAttr = ::mlir::triton::gpu::MACAMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

#define TT_MODE 2
using CacheMap = std::map<llvm::SmallVector<int, 2>, llvm::SmallVector<Value>>;

namespace mlir {
namespace MACA {

// Get a warpId for M axis.
static Value getWarpM(ConversionPatternRewriter &rewriter, const Location &loc,
                      int M, unsigned elemsM, ArrayRef<unsigned int> wpt,
                      Value warp, bool enableLdsTrans) {
  auto threadShapeM = getMmaThreadShape(enableLdsTrans)[0];
  return urem(urem(warp, i32_val(wpt[0])),
              i32_val(M / (threadShapeM * elemsM)));
}

// Get a warpId for N axis.
static Value getWarpN(ConversionPatternRewriter &rewriter, const Location &loc,
                      int N, unsigned elemsN, ArrayRef<unsigned int> wpt,
                      Value warp, bool enableLdsTrans) {
  auto threadShapeN = getMmaThreadShape(enableLdsTrans)[1];
  Value warpMN = udiv(warp, i32_val(wpt[0]));
  return urem(urem(warpMN, i32_val(wpt[1])),
              i32_val(N / (threadShapeN * elemsN)));
}

Value getThreadId(ConversionPatternRewriter &rewriter, Location loc) {
  auto tid =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);
}

// Data loader for mma maca instruction.
class MACAMMASmemLoader {
public:
  // loadA,rowMajor order=(1,0)
  // kOrder=1,threadShape=(16,4),matShape=(16,4),tileShape=(16,16)
  // smemStrides=(16,1) for one mma smemStrides=(4,1)
  //  TODO: threadShape = matShape, so matShape to be deprecated in MACA.
  MACAMMASmemLoader(int wpt, ArrayRef<uint32_t> order, uint32_t kOrder,
                    ArrayRef<Value> smemStrides, ArrayRef<int64_t> tileShape,
                    ArrayRef<int> threadShape, int perPhase, int maxPhase,
                    int elemBytes, bool enSmIdxCache, bool enSmIndexOpt,
                    ConversionPatternRewriter &rewriter,
                    const LLVMTypeConverter *typeConverter, const Location &loc,
                    ArrayRef<unsigned> elemsPerThread = {1, 1},
                    bool enableLdsTrans = false)
      : order(order.begin(), order.end()), kOrder(kOrder),
        tileShape(tileShape.begin(), tileShape.end()),
        threadShape(threadShape.begin(), threadShape.end()), perPhase(perPhase),
        maxPhase(maxPhase), elemBytes(elemBytes), enSmIdxCache(enSmIdxCache),
        enSmIndexOpt(enSmIndexOpt), rewriter(rewriter), loc(loc), wpt(wpt),
        elemsPerThread(elemsPerThread), enableLdsTrans(enableLdsTrans),
        ctx(rewriter.getContext()) {
    sMatShape = threadShape[order[1]];

    // For A, sStride = shape_k; for B, sStride = shape_n;
    sStride = smemStrides[order[1]];

    // rule: k must be the fast-changing axis.
    // if the mat is not continious in the K direction, would need transpose
    // e.g.
    // Mat A, kOrder=1, if it's row maj(order{1,0}, needTrans=False
    // Mat A, kOrder=1, if it's col maj(order{0,1}, needTrans=True
    // Mat B, kOrder=0, if it's row maj(order{1,0}, needTrans=True
    // Mat B, kOrder=0, if it's col maj(order{0,1}, needTrans=False
    needTrans = kOrder != order[0];

    // Each CTA, the warps is arranged as [1xwpt] if not transposed,
    // otherwise [wptx1], and each warp will perform a mma.
    // for Mat A, if it's row maj, numPtrs arrange in K direction
    // for Mat A, if it's col maj, numPtrs arrange in M direction
    numPtrs =
        tileShape[order[0]] / (needTrans ? wpt : 1) / threadShape[order[0]];
    // numPtrs = numPtrs / 4;

    int loadStrideInMat[2];
    loadStrideInMat[kOrder] =
        1; // threadShape[kOrder] / matShape[kOrder], always 1
    loadStrideInMat[kOrder ^ 1] =
        wpt; // wpt * threadShape[kOrder ^ 1] / matShape[kOrder ^ 1]);

    pLoadStrideInMat = loadStrideInMat[order[0]];
    sMatStride = loadStrideInMat[order[1]];
    // loadStrideInMat[order[1]] / (threadShape[order[1]] / matShape[order[1]]);

    // Each matArr contains warpOffStride matrices.
    matArrStride = kOrder == 1 ? 1 : wpt;
    warpOffStride = 1; // threadShape[kOrder ^ 1] / matShape[kOrder ^ 1];
  }

  int getNumPtrs() const { return numPtrs; }

  // TODO(MACA): to be deprecated.
  // Compute matrix offsets.
  // matrix layout coordinate inside a CTA, if wpt=3 for B(kOrder=0) is
  // |0 1 2| for A is
  // |0|
  // |1|
  // |2|
  // if each warp processes 2 mma, for B is |0 1 2 0 1 2|,
  // for A is
  // |0|
  // |1|
  // |2|
  // |0|
  // |1|
  // |2|
  SmallVector<Value> computeMatOffs(Value warpId, Value lane,
                                    Value cSwizzleOffset, int elemsPerThread,
                                    int curRepMN = 0, int curRepK = 0) {
    if (!needTrans) {
      // threads into 16*4 matrix
      // threadShape[0] is M, threadShape[1] is K
      Value c = urem(lane, i32_val(threadShape[order[1]])); // 16 rows
      Value s = udiv(lane, i32_val(threadShape[order[1]])); // 4 cols

      // calculate warp offset of the CTA tile.
      Value warpOff = mul(warpId, i32_val(warpOffStride)); // warpOffStride = 1

      // calculate column offset of the CTA tile.
      Value cOff = add(mul(warpOff, i32_val(threadShape[order[1]])), c);
      SmallVector<Value> offs(numPtrs);
      for (int i = 0; i < numPtrs; ++i) {
        // calculate row offset of the CTA tile.
        Value sOff =
            mul(add(s, i32_val(i * pLoadStrideInMat * threadShape[order[0]] /
                               elemsPerThread)),
                i32_val(elemsPerThread));

        // calculate ptr offset of the CTA tile.
        offs[i] = add(
            sOff,
            mul(cOff,
                sStride)); // sStride -> tile K -> offs[i] = col + row x tile_K
      }
      return offs;
    } else {
      // threads into 4*16 matrix
      // threadShape[0] is K, threadShape[1] is N
      Value c = udiv(lane, i32_val(threadShape[order[0]])); // 4 rows
      Value s = urem(lane, i32_val(threadShape[order[0]])); // 16 cols

      // calculate warp offset of the CTA tile.
      Value warpOff = mul(warpId, i32_val(warpOffStride)); // warpOffStride = 1
      SmallVector<Value> offs(numPtrs);
      for (int i = 0; i < numPtrs; ++i) {
        // calculate mat row offset of the CTA tile.
        Value sMatOff = add(
            warpOff, i32_val(i * pLoadStrideInMat)); // pLoadStrideInMat = wpt;
        // calculate row offset of the CTA tile.
        Value sOff = add(s, mul(sMatOff, i32_val(threadShape[order[0]])));
        // calculate offset of the CTA tile.
        offs[i] = add(sOff, mul(sStride, mul(c, i32_val(elemsPerThread))));
      }
      return offs;
    }
  }

  // TODO(liuyuxin) Support other types fp16, tf32.
  llvm::SmallVector<Value>
  computeTOffsets(Value warpOff, Value lane, Value cSwizzleOffset, int a, int b,
                  SmallVector<int> numRep, int minVec, int outVec, Value tensor,
                  ConversionPatternRewriter &rewriter, bool isA,
                  ArrayRef<unsigned> elemsPerThread,
                  METAX::IndexCacheInfoSm indexCacheInfoSm, CacheMap &cacheMap,
                  MmaEncodingAttr mmaLayout, Location loc) {
    if (elemBytes == 4 || elemBytes == 2 || elemBytes == 1) {
      int strideInt = -1; // default value -1, original mode
      int offsetInt = -1; // default value -1, original mode

      bool isNotNeedSwizzleOffset =
          std::getenv("TRITON_DISABLE_SWIZZLE") != nullptr || maxPhase == 1;
      auto elemsStride = isA ? mmaLayout.getElementsStride()[0]
                             : mmaLayout.getElementsStride()[1];

      // get stride int type
      strideInt = getStrideIntType(sStride, rewriter);

      // get swizzle offset int type
      offsetInt = swizzleOffsetIntType(cSwizzleOffset, isNotNeedSwizzleOffset,
                                       rewriter);

      MemDescType tensorTy = cast<MemDescType>(tensor.getType());
      auto sharedLayout = cast<SharedEncodingAttr>(tensorTy.getEncoding());
      // for row maj, order is 1, 0; for col maj, it is 0, 1
      auto order = sharedLayout.getOrder();

      // A:
      //    order=[1, 0], needTrans=False, cache along the m
      //    order=[0, 1], needTrans=True, cache along the k
      // B:
      //    order=[1, 0], needTrans=True, cache along the k
      //    order=[0, 1], needTrans=False, cache along the n
      bool enableIndexOpt =
          enSmIndexOpt && !enableLdsTrans && strideInt > -1 && offsetInt > -1 &&
          ((isA && order[0] == 1 && a > 0) || (isA && order[0] == 0 && b > 0) ||
           (!isA && order[0] == 1 && a > 0) ||
           (!isA && order[0] == 0 && b > 0));

      if (enSmIdxCache == false) {
        return computeTMatOffs(warpOff, lane, cSwizzleOffset, a, b, minVec,
                               outVec, strideInt, offsetInt, cacheMap,
                               enableIndexOpt, elemsStride);
      } else {
        if (strideInt < 0) {
          debug_print << "No cached strideInt, execute origin mode.\n";
          return computeTMatOffs(warpOff, lane, cSwizzleOffset, a, b, minVec,
                                 outVec, strideInt, offsetInt, cacheMap,
                                 enableIndexOpt, elemsStride);
        }

        if (!isNotNeedSwizzleOffset && offsetInt < 0) {
          debug_print << "No cached offsetInt, execute origin mode.\n";
          return computeTMatOffs(warpOff, lane, cSwizzleOffset, a, b, minVec,
                                 outVec, strideInt, offsetInt, cacheMap,
                                 enableIndexOpt, elemsStride);
        }

        Attribute srcEncoding = tensorTy.getEncoding();
        Type elementType = tensorTy.getElementType();
        const int inVec = elemsPerThread[order[0]];

        // Joined numRep because of chain-dot.
        SmallVector<int64_t> shapeKey{a,         b,         strideInt,
                                      offsetInt, inVec,     outVec,
                                      numRep[0], numRep[1], numRep[2]};
        for (int i = 0; i < elemsPerThread.size(); i++) {
          shapeKey.push_back(elemsPerThread[i]);
        }

        auto wpts = mmaLayout.getWarpsPerCTA();
        for (int i = 0; i < wpts.size(); i++) {
          shapeKey.push_back(wpts[i]);
        }

        auto keyRank = mlir::triton::MemDescType::get(
            shapeKey, elementType, srcEncoding, /*mutable*/ true);
        mlir::triton::METAX::IndexCacheKeyT key =
            std::make_pair(srcEncoding, keyRank);
        auto cache = indexCacheInfoSm.indexCache;
        auto insertPt = indexCacheInfoSm.indexInsertPoint;

        if (cache && cache->count(key) > 0) {
          debug_print << "index enter cache.\n";
          return cache->lookup(key);
        }

        // TODO:(yxwang) It can be placed in a better position, such as after
        // 'ldg op'.
        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        if (cache) {
          restoreInsertionPointIfSet(insertPt, rewriter);
        }

        Value _64 = i32_val(64);
        Value threadId = getThreadId(rewriter, loc);
        Value lane_cache = urem(threadId, _64);
        Value warp_cache = udiv(threadId, _64);

        SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                                   tensorTy.getShape().end());

        Value warp_cache_mn;
        if (isA) {
          warp_cache_mn = getWarpM(rewriter, loc, shape[0], elemsPerThread[0],
                                   wpts, warp_cache, enableLdsTrans);
          auto threadShapeM = getMmaThreadShape(enableLdsTrans)[0];
          auto matrixShapeM = threadShapeM * elemsPerThread[0];
          wpt = std::min<int>(wpts[0], shape[0] / matrixShapeM);
        } else {
          warp_cache_mn = getWarpN(rewriter, loc, shape[1], elemsPerThread[1],
                                   wpts, warp_cache, enableLdsTrans);
          auto threadShapeN = getMmaThreadShape(enableLdsTrans)[1];
          auto matrixShapeN = threadShapeN * elemsPerThread[1];
          wpt = std::min<int>(wpts[1], shape[1] / matrixShapeN);
        }

        auto result = computeTMatOffs(
            warp_cache_mn, lane_cache, cSwizzleOffset, a, b, minVec, outVec,
            strideInt, offsetInt, cacheMap, enableIndexOpt, elemsStride);

        if (cache) {
          cache->insert(std::make_pair(key, result));
          *insertPt = rewriter.saveInsertionPoint();
        }
        return result;
      }
    } else {
      llvm::report_fatal_error("Invalid smem load config");
    }
    return {};
  }

  // swizzle:
  //    compute phase = (row // perPhase) % maxPhase
  //    rowOff = row
  //    colOff = colOffSwizzled + colOffOrdered
  //       colOffSwizzled = ((col // outVec) ^ phase) * outVec
  //       colOffOrdered = (col % outVec) // minVec * minVec
  SmallVector<Value> computeTMatOffs(Value warpId, Value lane,
                                     Value cSwizzleOffset, int a, int b,
                                     int minVec, int outVec, int strideInt,
                                     int offsetInt, CacheMap &cacheMap,
                                     bool enableIndexOpt,
                                     unsigned elemsStride) {

    bool isNotNeedSwizzleOffset =
        std::getenv("TRITON_DISABLE_SWIZZLE") != nullptr || maxPhase == 1;
    bool enableElemsIndexOpt =
        enSmIndexOpt && strideInt > -1 && offsetInt > -1 && !enableLdsTrans;

    // if kOrder == 1, a is m, b is k;
    if (kOrder == 1) {
      // threads into 16*4 matrix
      // threadShape[0] is M, threadShape[1] is K
      // elemsPerThread[0] is elemsM, elemsPerThread[1] is elemsK.
      Value c = urem(lane, i32_val(threadShape[0]));
      Value s = udiv(lane, i32_val(threadShape[0]));

      // calculate warp offset of the CTA tile.
      Value warpOff = mul(warpId, i32_val(warpOffStride));

      Value aRowStride;
      Value aColStride;
      Value cSwiOffset;
      Value aCol;
      Value aRow = mul(
          add(c, mul(add(warpOff, i32_val(wpt * a)), i32_val(threadShape[0]))),
          i32_val(elemsPerThread[0]));
      if (enableLdsTrans) {
        assert(elemsStride <= threadShape[1]);
        Value blockOff = i32_val(b * threadShape[1] * elemsPerThread[1]);
        aCol = add(mul(udiv(s, i32_val(elemsStride)),
                       i32_val(elemsStride * elemsPerThread[1])),
                   urem(s, i32_val(elemsStride)));
        aCol = add(blockOff, aCol);
      } else {
        aCol = mul(add(s, i32_val(b * threadShape[1])),
                   i32_val(elemsPerThread[1]));
      }

      if (needTrans) {
        if (enableIndexOpt) {
          // cache along k
          int off = b * getMmaThreadShape(enableLdsTrans)[2] *
                    elemsPerThread[1] * strideInt;
          auto baseOff = cacheMap[{a, 0}];
          for (int i = 0; i < elemsPerThread[1]; i++) {
            baseOff[i] = add(baseOff[i], i32_val(off));
          }
          return baseOff;
        } else {
          aRowStride = i32_val(1);

          if (enSmIdxCache == false || strideInt < 0 ||
              (!isNotNeedSwizzleOffset && offsetInt < 0)) {
            aColStride = sStride;
            cSwiOffset = cSwizzleOffset;
          } else {
            aColStride = i32_val(strideInt);
            cSwiOffset = i32_val(offsetInt);
          }

          SmallVector<Value> offs(elemsPerThread[1]);
          for (int i = 0; i < elemsPerThread[1]; i++) {
            if (enableElemsIndexOpt && (i % perPhase != 0)) {
              int baseIdx = (i / perPhase) * perPhase;
              int off = (i - baseIdx) * strideInt;
              offs[i] = add(offs[baseIdx], i32_val(off));
            } else {
              if (isNotNeedSwizzleOffset) {
                Value aCol_ = add(aCol, i32_val(i * elemsStride));
                offs[i] = add(mul(aRow, aRowStride), mul(aCol_, aColStride));
              } else {
                Value aCol_ = add(aCol, i32_val(i * elemsStride));
                Value phase =
                    urem(udiv(aCol_, i32_val(perPhase)), i32_val(maxPhase));
                Value idxRow = add(cSwiOffset, aRow);
                Value OffSwizzled = xor_(udiv(idxRow, i32_val(outVec)), phase);
                OffSwizzled = mul(OffSwizzled, i32_val(outVec));
                if (outVec == minVec) {
                  Value rowOff = OffSwizzled;
                  offs[i] =
                      add(mul(rowOff, aRowStride), mul(aCol_, aColStride));
                } else {
                  Value OffOrdered = urem(idxRow, i32_val(outVec));
                  Value rowOff = add(OffSwizzled, OffOrdered);
                  offs[i] =
                      add(mul(rowOff, aRowStride), mul(aCol_, aColStride));
                }
              }
            }
          }
          cacheMap.insert({{a, b}, offs});
          return offs;
        }
      } else {
        if (enableIndexOpt) {
          // cache along m
          int off = a * getMmaThreadShape(enableLdsTrans)[0] *
                    elemsPerThread[0] * strideInt * wpt;
          auto baseOff = cacheMap[{0, b}];
          for (int i = 0; i < elemsPerThread[0]; i++) {
            baseOff[i] = add(baseOff[i], i32_val(off));
          }
          return baseOff;
        } else {
          Value aColStride = i32_val(1);

          if (enSmIdxCache == false || strideInt < 0 ||
              (!isNotNeedSwizzleOffset && offsetInt < 0)) {
            aRowStride = sStride;
            cSwiOffset = cSwizzleOffset;
          } else {
            aRowStride = i32_val(strideInt);
            cSwiOffset = i32_val(offsetInt);
          }

          SmallVector<Value> offs(elemsPerThread[0]);
          for (int i = 0; i < elemsPerThread[0]; i++) {
            if (enableElemsIndexOpt && (i % perPhase != 0)) {
              int baseIdx = (i / perPhase) * perPhase;
              int off = (i - baseIdx) * strideInt;
              offs[i] = add(offs[baseIdx], i32_val(off));
            } else {
              if (isNotNeedSwizzleOffset) {
                Value aRow_ = add(aRow, i32_val(i));
                offs[i] = add(mul(aRow_, aRowStride), mul(aCol, aColStride));
              } else {
                Value aRow_ = add(aRow, i32_val(i));
                Value phase =
                    urem(udiv(aRow_, i32_val(perPhase)), i32_val(maxPhase));
                Value idxCol = add(cSwiOffset, aCol);
                Value OffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
                OffSwizzled = mul(OffSwizzled, i32_val(outVec));
                if (outVec == minVec) {
                  Value colOff = OffSwizzled;
                  offs[i] =
                      add(mul(aRow_, aRowStride), mul(colOff, aColStride));
                } else {
                  Value OffOrdered = urem(idxCol, i32_val(outVec));
                  Value colOff = add(OffSwizzled, OffOrdered);
                  offs[i] =
                      add(mul(aRow_, aRowStride), mul(colOff, aColStride));
                }
              }
            }
          }
          cacheMap.insert({{a, b}, offs});
          return offs;
        }
      }
    } else {
      // if kOrder == 0, a is k, b is n;
      // threads into 4*16 matrix
      // threadShape[0] is K, threadShape[1] is N
      // elemsPerThread[0] is elemsK, elemsPerThread[1] is elemsN.
      Value c = udiv(lane, i32_val(threadShape[1]));
      Value s = urem(lane, i32_val(threadShape[1]));

      // calculate warp offset of the CTA tile.
      Value warpOff = mul(warpId, i32_val(warpOffStride));
      Value bRow;
      if (enableLdsTrans) {
        assert(elemsStride <= threadShape[0]);
        Value blockOff = i32_val(a * threadShape[0] * elemsPerThread[0]);
        bRow = add(mul(udiv(c, i32_val(elemsStride)),
                       i32_val(elemsStride * elemsPerThread[0])),
                   urem(c, i32_val(elemsStride)));
        bRow = add(blockOff, bRow);
      } else {
        bRow = mul(add(c, i32_val(a * threadShape[0])),
                   i32_val(elemsPerThread[0]));
      }
      Value bCol = mul(
          add(s, mul(add(warpOff, i32_val(wpt * b)), i32_val(threadShape[1]))),
          i32_val(elemsPerThread[1]));
      Value bRowStride;
      Value bColStride;
      Value cSwiOffset;

      if (needTrans) {
        if (enableIndexOpt) {
          // cache along k
          int off = a * getMmaThreadShape(enableLdsTrans)[2] *
                    elemsPerThread[0] * strideInt;
          auto baseOff = cacheMap[{0, b}];
          for (int i = 0; i < elemsPerThread[0]; i++) {
            baseOff[i] = add(baseOff[i], i32_val(off));
          }
          return baseOff;
        } else {
          bColStride = i32_val(1);

          if (enSmIdxCache == false || strideInt < 0 ||
              (!isNotNeedSwizzleOffset && offsetInt < 0)) {
            bRowStride = sStride;
            cSwiOffset = cSwizzleOffset;
          } else {
            bRowStride = i32_val(strideInt);
            cSwiOffset = i32_val(offsetInt);
          }

          SmallVector<Value> offs(elemsPerThread[0]);
          for (int i = 0; i < elemsPerThread[0]; i++) {
            if (enableElemsIndexOpt && (i % perPhase != 0)) {
              // perPhase = std::max(tk, perPhase) which defined in
              // TritonGPUAttrDefs.td, but if we manually modify the
              // shared_encoding (using pass like
              // TRITON_ENABLE_MACA_MERGE_EQUAL_SHARED_LAYOUT) that may cause
              // perPhase < tk, in this case, we should cal the off according to
              // perPhase
              int baseIdx = (i / perPhase) * perPhase;
              int off = (i - baseIdx) * strideInt;
              offs[i] = add(offs[baseIdx], i32_val(off));
            } else {
              if (isNotNeedSwizzleOffset) {
                Value bRow_ = add(bRow, i32_val(i * elemsStride));
                offs[i] = add(mul(bRow_, bRowStride), mul(bCol, bColStride));
              } else {
                Value bRow_ = add(bRow, i32_val(i * elemsStride));
                Value phase =
                    urem(udiv(bRow_, i32_val(perPhase)), i32_val(maxPhase));
                Value idxCol = add(cSwiOffset, bCol);
                Value OffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
                OffSwizzled = mul(OffSwizzled, i32_val(outVec));
                if (outVec == minVec) {
                  Value colOff = OffSwizzled;
                  offs[i] =
                      add(mul(bRow_, bRowStride), mul(colOff, bColStride));
                } else {
                  Value OffOrdered = urem(idxCol, i32_val(outVec));
                  Value colOff = add(OffSwizzled, OffOrdered);
                  offs[i] =
                      add(mul(bRow_, bRowStride), mul(colOff, bColStride));
                }
              }
            }
          }
          cacheMap.insert({{a, b}, offs});
          return offs;
        }
      } else {
        if (enableIndexOpt) {
          // cache along n
          int off = b * getMmaThreadShape(enableLdsTrans)[1] *
                    elemsPerThread[1] * strideInt * wpt;
          auto baseOff = cacheMap[{a, 0}];
          for (int i = 0; i < elemsPerThread[1]; i++) {
            baseOff[i] = add(baseOff[i], i32_val(off));
          }
          return baseOff;
        } else {
          Value bRowStride = i32_val(1);

          if (enSmIdxCache == false || strideInt < 0 ||
              (!isNotNeedSwizzleOffset && offsetInt < 0)) {
            bColStride = sStride;
            cSwiOffset = cSwizzleOffset;
          } else {
            bColStride = i32_val(strideInt);
            cSwiOffset = i32_val(offsetInt);
          }

          SmallVector<Value> offs(elemsPerThread[1]);
          for (int i = 0; i < elemsPerThread[1]; i++) {
            if (enableElemsIndexOpt && (i % perPhase != 0)) {
              int baseIdx = (i / perPhase) * perPhase;
              int off = (i - baseIdx) * strideInt;
              offs[i] = add(offs[baseIdx], i32_val(off));
            } else {
              if (isNotNeedSwizzleOffset) {
                Value bCol_ = add(bCol, i32_val(i));
                offs[i] = add(mul(bRow, bRowStride), mul(bCol_, bColStride));
              } else {
                Value bCol_ = add(bCol, i32_val(i));
                Value phase =
                    urem(udiv(bCol_, i32_val(perPhase)), i32_val(maxPhase));
                Value idxRow = add(cSwiOffset, bRow);
                Value OffSwizzled = xor_(udiv(idxRow, i32_val(outVec)), phase);
                OffSwizzled = mul(OffSwizzled, i32_val(outVec));
                if (outVec == minVec) {
                  Value rowOff = OffSwizzled;
                  offs[i] =
                      add(mul(rowOff, bRowStride), mul(bCol_, bColStride));
                } else {
                  Value OffOrdered = urem(idxRow, i32_val(outVec));
                  Value rowOff = add(OffSwizzled, OffOrdered);
                  offs[i] =
                      add(mul(rowOff, bRowStride), mul(bCol_, bColStride));
                }
              }
            }
          }
          cacheMap.insert({{a, b}, offs});
          return offs;
        }
      }
    }
    return {};
  }

  // Load n matrix and returns vec<1> element.
  SmallVector<Value> loadXN(triton::gpu::LocalLoadOp op, ArrayRef<Value> ptrs,
                            int elemBytes, Type type, Value const_offset,
                            MmaEncodingAttr mmaLayout) const {
    std::string lds_intrinsic = "llvm.mxc.lds";
    std::string lds_trans_b16_intrinsic = "llvm.mxc.load.shared.trans.4x16";
    std::string lds_trans_b8_intrinsic = "llvm.mxc.load.shared.trans.8x16";
    if (elemBytes == 8) {
      Type elemType;
      if (type.isF64()) {
        elemType = f64_ty;
      } else if (type.isSignlessInteger(64)) {
        elemType = i64_ty;
      } else {
        assert(false && "Invalid smem load");
      }
      SmallVector<Value> regs(elemsPerThread[0] * elemsPerThread[1]);
      assert(ptrs.size() == elemsPerThread[order[1]] && "Invalid smem load");
      Type xnTy = vec_ty(elemType, elemsPerThread[order[0]]);
#ifndef USE_MACA_OPAQUE_PTR
      Type xnPtrTy = ptr_ty(xnTy, 3);
#else
      Type xnPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
      appendIntrinsicModifer(lds_intrinsic, elemsPerThread[order[0]], elemType);
      for (int j = 0; j < elemsPerThread[order[1]]; ++j) {
        Value elemXN = undef(xnTy);
        if (op.getIsConstantOffs()) {
#ifndef USE_MACA_OPAQUE_PTR
          auto elemXNPtr =
              bitcast(gep(ptr_ty(elemType, 3), elemType, ptrs[j], const_offset),
                      xnPtrTy);
#else
          auto elemXNPtr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                       elemType, ptrs[j], const_offset),
                                   xnPtrTy);
#endif
          elemXN = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
              rewriter, loc, op, lds_intrinsic, xnTy, {elemXNPtr});
        } else {
          elemXN = load(xnTy, bitcast(ptrs[j], xnPtrTy));
        }
        for (int i = 0; i < elemsPerThread[order[0]]; ++i) {
          regs[j * elemsPerThread[order[0]] + i] =
              extract_element(elemXN, i32_val(i));
        }
      }
      return regs;
    } else if (elemBytes == 4) {
      SmallVector<Value> regs(elemsPerThread[0] * elemsPerThread[1]);
      assert(ptrs.size() == elemsPerThread[order[1]] && "Invalid smem load");
      Type fp32xnTy = vec_ty(type::f32Ty(ctx), elemsPerThread[order[0]]);
#ifndef USE_MACA_OPAQUE_PTR
      Type fp32xnPtrTy = ptr_ty(fp32xnTy, 3);
#else
      Type fp32xnPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
      appendIntrinsicModifer(lds_intrinsic, elemsPerThread[order[0]],
                             type::f32Ty(ctx));
      for (int j = 0; j < elemsPerThread[order[1]]; ++j) {
        Value elemXN = undef(fp32xnTy);
        if (op.getIsConstantOffs()) {
#ifndef USE_MACA_OPAQUE_PTR
          auto elemXNPtr = bitcast(gep(ptr_ty(type::f32Ty(ctx), 3),
                                       type::f32Ty(ctx), ptrs[j], const_offset),
                                   fp32xnPtrTy);
#else
          auto elemXNPtr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                       type::f32Ty(ctx), ptrs[j], const_offset),
                                   fp32xnPtrTy);
#endif
          elemXN = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
              rewriter, loc, op, lds_intrinsic, fp32xnTy, {elemXNPtr});
        } else {
          elemXN = load(fp32xnTy, bitcast(ptrs[j], fp32xnPtrTy));
        }
        for (int i = 0; i < elemsPerThread[order[0]]; ++i) {
          regs[j * elemsPerThread[order[0]] + i] =
              extract_element(elemXN, i32_val(i));
        }
      }
      return regs;
    } else if (elemBytes == 2) {
      // Type elemType = type::f16Ty(ctx);
      Type elemType;
      if (type.isF16()) {
        elemType = type::f16Ty(ctx);
      } else if (type.isBF16() || type.isSignlessInteger(16)) {
        elemType = type::i16Ty(ctx);
      } else if (type.isF32()) { // TF32
        elemType = type::f32Ty(ctx);
      } else {
        assert(false && "Invalid smem load");
      }
      SmallVector<Value> regs(elemsPerThread[0] * elemsPerThread[1]);
      Type fp16x4Ty = vec_ty(elemType, 4);
      Type int16x4Ty = vec_ty(type::i16Ty(ctx), 4);
#ifdef USE_MACA_OPAQUE_PTR
      Type int16x4PtrTy = ptr_ty(rewriter.getContext(), 3);
#else
      Type int16x4PtrTy = ptr_ty(rewriter.getContext(), int16x4Ty, 3);
#endif
      if (enableLdsTrans) {
        auto elemsStride = kOrder == 1 ? mmaLayout.getElementsStride()[0]
                                       : mmaLayout.getElementsStride()[1];
        for (int j = 0; j < elemsPerThread[order[1]]; j++) {
          for (int i = 0; i < elemsPerThread[order[0]] / elemsStride; i++) {
            Value elemInt16X4 = undef(int16x4Ty);
            Value elemFp16X4 = undef(fp16x4Ty);
            if (op.getIsConstantOffs()) {
              Value offset = add(const_offset, i32_val(elemsStride * i));
#ifdef USE_MACA_OPAQUE_PTR
              auto elemX4Ptr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                           elemType, ptrs[j], offset),
                                       int16x4PtrTy);
#else
              auto elemX4Ptr =
                  bitcast(gep(ptr_ty(elemType, 3), elemType, ptrs[j], offset),
                          int16x4PtrTy);
#endif
              elemInt16X4 =
                  mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                      rewriter, loc, op, lds_trans_b16_intrinsic, int16x4Ty,
                      {elemX4Ptr});
            } else {
              Value offset = i32_val(elemsStride * i);
#ifdef USE_MACA_OPAQUE_PTR
              auto elemX4Ptr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                           elemType, ptrs[j], offset),
                                       int16x4PtrTy);
#else
              auto elemX4Ptr =
                  bitcast(gep(ptr_ty(elemType, 3), elemType, ptrs[j], offset),
                          int16x4PtrTy);
#endif
              elemInt16X4 =
                  mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                      rewriter, loc, op, lds_trans_b16_intrinsic, int16x4Ty,
                      {elemX4Ptr});
            }
            elemFp16X4 = bitcast(elemInt16X4, fp16x4Ty);
            for (int idx = 0; idx < elemsStride; idx++)
              regs[(j * elemsStride + idx) *
                       (elemsPerThread[order[0]] / elemsStride) +
                   i] = extract_element(elemFp16X4, i32_val(idx));
          }
        }
      } else {
        bool needPerm = false;
        if (elemsPerThread[order[1]] < 2 || elemsPerThread[order[0]] < 2) {
          needPerm = false;
        } else {
          // B matrix Row maj and A matrix Col maj
          if ((kOrder == 0 && order[0] == 1) ||
              (kOrder == 1 && order[0] == 0)) {
            needPerm = true;
          }
        }
        assert(ptrs.size() == elemsPerThread[order[1]] &&
               "Invalid smem load"); // A: M, B:K
        if (!needPerm) {
          Type f16xnTy = vec_ty(elemType, elemsPerThread[order[0]]); // A:K, B:N
#ifdef USE_MACA_OPAQUE_PTR
          Type f16xnPtrTy = ptr_ty(rewriter.getContext(), 3);
#else
          Type f16xnPtrTy = ptr_ty(rewriter.getContext(), f16xnTy, 3);
#endif
          appendIntrinsicModifer(lds_intrinsic, elemsPerThread[order[0]],
                                 elemType);
          for (int j = 0; j < elemsPerThread[order[1]]; ++j) {
            Value elemXN = undef(f16xnTy);
            if (op.getIsConstantOffs()) {
#ifdef USE_MACA_OPAQUE_PTR
              auto elemXNPtr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                           elemType, ptrs[j], const_offset),
                                       f16xnPtrTy);
#else
              auto elemXNPtr = bitcast(
                  gep(ptr_ty(elemType, 3), elemType, ptrs[j], const_offset),
                  f16xnPtrTy);
#endif
              elemXN = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                  rewriter, loc, op, lds_intrinsic, f16xnTy, {elemXNPtr});
            } else {
              elemXN = load(f16xnTy, bitcast(ptrs[j], f16xnPtrTy));
            }
            for (int i = 0; i < elemsPerThread[order[0]]; ++i) {
              regs[j * elemsPerThread[order[0]] + i] =
                  extract_element(elemXN, i32_val(i));
            }
          }
        } else {
          Type fp16xnTy =
              vec_ty(elemType, elemsPerThread[order[0]]); // A:K, B:N
          Type fp16x2Ty = vec_ty(elemType, 2);
          Type int32xnTy = vec_ty(type::i32Ty(ctx),
                                  elemsPerThread[order[0]] / 2); // A:K, B:N
          Type int32Ty = type::i32Ty(ctx);
#ifdef USE_MACA_OPAQUE_PTR
          Type fp16xnPtrTy = ptr_ty(rewriter.getContext(), 3);
          Type int32xnPtrTy = ptr_ty(rewriter.getContext(), 3);
#else
          Type fp16xnPtrTy = ptr_ty(rewriter.getContext(), fp16xnTy, 3);
          Type int32xnPtrTy = ptr_ty(rewriter.getContext(), int32xnTy, 3);
#endif
          appendIntrinsicModifer(lds_intrinsic, elemsPerThread[order[0]] / 2,
                                 type::i32Ty(ctx));

          if (op.getMmaMode() == TT_MODE && op.getIsConstantOffs()) {
            for (int j = 0; j < elemsPerThread[order[1]] / 2; j += 2) {
              Value halfValuex2Front = undef(fp16x2Ty);
              Value halfValuex2Back = undef(fp16x2Ty);

              Type int32xnTy = vec_ty(type::i32Ty(ctx),
                                      elemsPerThread[order[0]] / 2); // A:K, B:N
              Type int32Ty = type::i32Ty(ctx);

              Value elemInt32 = undef(int32xnPtrTy);
              Value elemInt32_st = undef(int32xnPtrTy);
              Value permValueInt32Front = undef(i32_ty);
              Value permValueInt32Back = undef(i32_ty);
#ifdef USE_MACA_OPAQUE_PTR
              Type elem_ptr_ty = ptr_ty(rewriter.getContext(), 3);
#else
              Type elem_ptr_ty = ptr_ty(rewriter.getContext(), elemType, 3);
#endif

              // first lds
              if (op.getIsConstantOffs()) {
                auto elemXNPtr =
                    bitcast(gep(elem_ptr_ty, elemType, ptrs[j], const_offset),
                            int32xnPtrTy);
                elemInt32 =
                    mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                        rewriter, loc, op, lds_intrinsic, int32xnTy,
                        {elemXNPtr});

                // To merge the LDS instruction, the original offset needs to be
                // divisible by 512. lds_b64 r2, r0, 0; and lds_b64 r3, r0, 512;
                // --->lds_st_b64x2 r2, r0, 0, 1;
                Value offset_st = add(const_offset, i32_val(512));
                auto elemXNPtr_st =
                    bitcast(gep(elem_ptr_ty, elemType, ptrs[j], offset_st),
                            int32xnPtrTy);
                elemInt32_st =
                    mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                        rewriter, loc, op, lds_intrinsic, int32xnTy,
                        {elemXNPtr_st});
              }

              // second lds
              Value elemInt32_2 = undef(int32xnPtrTy);
              Value elemInt32_2_st = undef(int32xnPtrTy);
              if (op.getIsConstantOffs()) {
                auto elemXNPtr = bitcast(
                    gep(elem_ptr_ty, elemType, ptrs[j + 1], const_offset),
                    int32xnPtrTy);
                elemInt32_2 =
                    mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                        rewriter, loc, op, lds_intrinsic, int32xnTy,
                        {elemXNPtr});

                // To merge the LDS instruction, the original offset needs to be
                // divisible by 512. lds_b64 r2, r0, 0; and lds_b64 r3, r0, 512;
                // --->lds_st_b64x2 r2, r0, 0, 1;
                Value offset_st = add(const_offset, i32_val(512));
                auto elemXNPtr_st_2 =
                    bitcast(gep(elem_ptr_ty, elemType, ptrs[j + 1], offset_st),
                            int32xnPtrTy);
                elemInt32_2_st =
                    mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                        rewriter, loc, op, lds_intrinsic, int32xnTy,
                        {elemXNPtr_st_2});
              }

              int halfOfElemsSize =
                  (elemsPerThread[order[0]] * elemsPerThread[order[1]]) / 2;
              for (int idx = 0; idx < elemsPerThread[order[0]] / 2; ++idx) {
                regs[4 * idx + 0 + j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32, i32_val(idx));
                regs[4 * idx + 1 + j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32_2, i32_val(idx));
                regs[4 * idx + 2 + j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32, i32_val(idx));
                regs[4 * idx + 3 + j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32_2, i32_val(idx));

                regs[4 * idx + 0 + halfOfElemsSize +
                     j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32_st, i32_val(idx));
                regs[4 * idx + 1 + halfOfElemsSize +
                     j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32_2_st, i32_val(idx));
                regs[4 * idx + 2 + halfOfElemsSize +
                     j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32_st, i32_val(idx));
                regs[4 * idx + 3 + halfOfElemsSize +
                     j * elemsPerThread[order[0]]] =
                    extract_element(elemInt32_2_st, i32_val(idx));
              }
            }
          } else {
#ifdef USE_MACA_OPAQUE_PTR
            Type elem_ptr_ty = ptr_ty(rewriter.getContext(), 3);
#else
            Type elem_ptr_ty = ptr_ty(rewriter.getContext(), elemType, 3);
#endif
            for (int j = 0; j < elemsPerThread[order[1]]; j += 2) {
              Value halfValuex2Front = undef(fp16x2Ty);
              Value halfValuex2Back = undef(fp16x2Ty);
              Value elemXN = undef(fp16xnTy);
              Value elemXN_2 = undef(fp16xnTy);

              Value elemInt32 = undef(int32xnPtrTy);
              Value permValueInt32Front = undef(i32_ty);
              Value permValueInt32Back = undef(i32_ty);
              // first lds
              if (op.getIsConstantOffs()) {
                auto elemXNPtr =
                    bitcast(gep(elem_ptr_ty, elemType, ptrs[j], const_offset),
                            int32xnPtrTy);
                elemInt32 =
                    mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                        rewriter, loc, op, lds_intrinsic, int32xnTy,
                        {elemXNPtr});
              } else {
                elemInt32 = load(int32xnTy, bitcast(ptrs[j], int32xnPtrTy));
              }

              elemXN = bitcast(elemInt32, fp16xnTy);

              // second lds
              Value elemInt32_2 = undef(int32xnPtrTy);

              if (op.getIsConstantOffs()) {
#ifndef USE_MACA_OPAQUE_PTR
                auto elemXNPtr = bitcast(gep(ptr_ty(elemType, 3), elemType,
                                             ptrs[j + 1], const_offset),
                                         int32xnPtrTy);
#else
                auto elemXNPtr =
                    bitcast(gep(ptr_ty(rewriter.getContext(), 3), elemType,
                                ptrs[j + 1], const_offset),
                            int32xnPtrTy);
#endif
                elemInt32_2 =
                    mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                        rewriter, loc, op, lds_intrinsic, int32xnTy,
                        {elemXNPtr});
              } else {
                elemInt32_2 =
                    load(int32xnTy, bitcast(ptrs[j + 1], int32xnPtrTy));
              }
              elemXN_2 = bitcast(elemInt32_2, fp16xnTy);

              Value offsetFront = i32_val(0x01000504);
              Value offsetBack = i32_val(0x03020706);

              SmallVector<Value> rangeValueFront(3);
              rangeValueFront[2] = offsetFront;
              SmallVector<Value> rangeValueBack(3);
              rangeValueBack[2] = offsetBack;

              std::string intrinsicPermName = "llvm.mxc.byte.perm";
              StringRef permName(intrinsicPermName);

              for (int idx = 0; idx < elemsPerThread[order[0]] / 2; ++idx) {
                rangeValueFront[0] = extract_element(elemInt32, i32_val(idx));
                rangeValueFront[1] = extract_element(elemInt32_2, i32_val(idx));
                ValueRange permValueRangeFront(rangeValueFront);

                rangeValueBack[0] = extract_element(elemInt32, i32_val(idx));
                rangeValueBack[1] = extract_element(elemInt32_2, i32_val(idx));
                ValueRange permValueRangeBack(rangeValueBack);

                permValueInt32Front = mlir::LLVM::createBuiltinFunc(
                    rewriter, loc, op, permName, int32Ty, permValueRangeFront);
                halfValuex2Front = bitcast(permValueInt32Front, fp16x2Ty);

                permValueInt32Back = mlir::LLVM::createBuiltinFunc(
                    rewriter, loc, op, permName, int32Ty, permValueRangeBack);
                halfValuex2Back = bitcast(permValueInt32Back, fp16x2Ty);

                regs[2 * idx + j * elemsPerThread[order[0]]] =
                    extract_element(halfValuex2Front, i32_val(0));
                regs[2 * idx + (j + 1) * elemsPerThread[order[0]]] =
                    extract_element(halfValuex2Front, i32_val(1));

                regs[2 * idx + 1 + j * elemsPerThread[order[0]]] =
                    extract_element(halfValuex2Back, i32_val(0));
                regs[2 * idx + 1 + (j + 1) * elemsPerThread[order[0]]] =
                    extract_element(halfValuex2Back, i32_val(1));
              }
            }
          }
        }
      }
      return regs;
    } else if (elemBytes == 1) {
      Type elemType;
      if (type.isSignlessInteger(8) || type.isFloat8E5M2() ||
          type.isFloat8E4M3FN()) {
        elemType = type::i8Ty(ctx);
      } else {
        assert(false && "Invalid smem load, only supported int8");
      }
      SmallVector<Value> regs(elemsPerThread[0] * elemsPerThread[1]);
      assert(ptrs.size() == elemsPerThread[order[1]] &&
             "Invalid smem load");                              // A: M, B:K
      Type i8xnTy = vec_ty(elemType, elemsPerThread[order[0]]); // A:K, B:N
      Type i8x8Ty = vec_ty(type::i8Ty(ctx), 8);
#ifndef USE_MACA_OPAQUE_PTR
      Type i8xnPtrTy = ptr_ty(rewriter.getContext(), i8xnTy, 3);
      Type i8x8PtrTy = ptr_ty(rewriter.getContext(), i8x8Ty, 3);
#else
      Type i8xnPtrTy = ptr_ty(rewriter.getContext(), 3);
      Type i8x8PtrTy = ptr_ty(rewriter.getContext(), 3);
#endif
      if (enableLdsTrans) {
        auto elemsStride = kOrder == 1 ? mmaLayout.getElementsStride()[0]
                                       : mmaLayout.getElementsStride()[1];
        for (int j = 0; j < elemsPerThread[order[1]]; j++) {
          for (int i = 0; i < elemsPerThread[order[0]] / elemsStride; i++) {
            Value elemI8X8 = undef(i8x8Ty);
            if (op.getIsConstantOffs()) {
              Value offset = add(const_offset, i32_val(elemsStride * i));
#ifndef USE_MACA_OPAQUE_PTR
              auto elemX8Ptr =
                  bitcast(gep(ptr_ty(elemType, 3), elemType, ptrs[j], offset),
                          i8x8PtrTy);
#else
              auto elemX8Ptr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                           elemType, ptrs[j], offset),
                                       i8x8PtrTy);
#endif
              elemI8X8 =
                  mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                      rewriter, loc, op, lds_trans_b8_intrinsic, i8x8Ty,
                      {elemX8Ptr});
            } else {
              Value offset = i32_val(elemsStride * i);
#ifndef USE_MACA_OPAQUE_PTR
              auto elemX8Ptr =
                  bitcast(gep(ptr_ty(elemType, 3), elemType, ptrs[j], offset),
                          i8x8PtrTy);
#else
              auto elemX8Ptr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                           elemType, ptrs[j], offset),
                                       i8x8PtrTy);
#endif
              elemI8X8 =
                  mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                      rewriter, loc, op, lds_trans_b8_intrinsic, i8x8Ty,
                      {elemX8Ptr});
            }
            for (int idx = 0; idx < elemsStride; idx++)
              regs[(j * elemsStride + idx) *
                       (elemsPerThread[order[0]] / elemsStride) +
                   i] = extract_element(elemI8X8, i32_val(idx));
          }
        }
      } else {
        appendIntrinsicModifer(lds_intrinsic, elemsPerThread[order[0]],
                               elemType);
        for (int j = 0; j < elemsPerThread[order[1]]; ++j) {
          Value elemXN = undef(i8xnTy);
          if (op.getIsConstantOffs()) {
#ifndef USE_MACA_OPAQUE_PTR
            auto elemXNPtr = bitcast(
                gep(ptr_ty(elemType, 3), elemType, ptrs[j], const_offset),
                i8xnPtrTy);
#else
            auto elemXNPtr = bitcast(gep(ptr_ty(rewriter.getContext(), 3),
                                         elemType, ptrs[j], const_offset),
                                     i8xnPtrTy);
#endif
            elemXN = mlir::LLVM::createBuiltinFunc<triton::gpu::LocalLoadOp>(
                rewriter, loc, op, lds_intrinsic, i8xnTy, {elemXNPtr});
          } else {
            elemXN = load(i8xnTy, bitcast(ptrs[j], i8xnPtrTy));
          }
          for (int i = 0; i < elemsPerThread[order[0]]; ++i) {
            regs[j * elemsPerThread[order[0]] + i] =
                extract_element(elemXN, i32_val(i));
          }
        }
      }
      return regs;
    } else {
      assert(false && "Invalid smem load");
      return {Value{}};
    }
  }

private:
  SmallVector<uint32_t> order;
  int kOrder;
  SmallVector<int64_t> tileShape;
  SmallVector<int> threadShape;
  SmallVector<unsigned> elemsPerThread;
  bool enableLdsTrans;
  int perPhase;
  int maxPhase;
  int elemBytes;
  bool enSmIdxCache;
  bool enSmIndexOpt;
  ConversionPatternRewriter &rewriter;
  const Location &loc;
  MLIRContext *ctx{};

  int sMatShape;

  Value sStride;

  bool needTrans;

  int numPtrs;

  int pLoadStrideInMat;
  int sMatStride;
  int wpt;

  int matArrStride;
  int warpOffStride;

private:
  void restoreInsertionPointIfSet(OpBuilder::InsertPoint *insertPt,
                                  ConversionPatternRewriter &rewriter) const {
    if (insertPt->isSet()) {
      rewriter.restoreInsertionPoint(*insertPt);
    } else {
      auto func =
          rewriter.getInsertionPoint()->getParentOfType<LLVM::LLVMFuncOp>();
      rewriter.setInsertionPointToStart(&func.getBody().front());
    }
  }
};

// Compose a map of Values to a LLVM::Struct.
// The layout is a list of Value with coordinate of (i,j), the order is as
// the follows:
// [
//  (0,0), (0,1), (1,0), (1,1), # i=0, j=0
//  (0,2), (0,3), (1,2), (1,3), # i=0, j=1
//  (0,4), (0,5), (1,4), (1,5), # i=0, j=2
//  ...
//  (2,0), (2,1), (3,0), (3,1), # i=1, j=0
//  (2,2), (2,3), (3,2), (3,3), # i=1, j=1
//  (2,4), (2,5), (3,4), (3,5), # i=1, j=2
//  ...
// ]
// i \in [0, n0) and j \in [0, n1)
// There should be \param n0 * \param n1 elements in the output Struct.
Value composeValuesToDotOperandLayoutStruct(
    const ValueTable &vals, int n0, int n1,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, int elemsPerThreadMN,
    int elemsPerThreadK) {
  std::vector<Value> elems;
  for (int m = 0; m < n0; ++m)
    for (int k = 0; k < n1; ++k)
      for (int j = 0; j < elemsPerThreadMN; ++j)
        for (int i = 0; i < elemsPerThreadK; ++i) {
          elems.push_back(
              vals.at({m * elemsPerThreadMN + j, k * elemsPerThreadK + i}));
        }

  assert(!elems.empty());

  Type elemTy = elems[0].getType();
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));
  auto result = packLLElements(loc, typeConverter, elems, rewriter, structTy);
  return result;
}

Value composeValuesToDotOperandLayoutStructForB(
    const SmallVector<Value> &valsForB, int n0, int n1,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, int elemsPerThreadMN,
    int elemsPerThreadK) {
  std::vector<Value> elems;
  for (int m = 0; m < n0; ++m)
    for (int k = 0; k < n1; ++k)
      for (int j = 0; j < elemsPerThreadK; ++j)
        for (int i = 0; i < elemsPerThreadMN; ++i) {
          elems.push_back(valsForB[j * elemsPerThreadMN + i]);
        }

  assert(!elems.empty());

  Type elemTy = elems[0].getType();
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));
  auto result = packLLElements(loc, typeConverter, elems, rewriter, structTy);

  return result;
}

void loadMatrixFn(triton::gpu::LocalLoadOp op, Value tensor,
                  const SharedMemoryObject &smemObj, MmaEncodingAttr mmaLayout,
                  int wpt, uint32_t kOrder, SmallVector<int> threadShape,
                  Value warpId, Value lane, ValueTable &vals,
                  SmallVector<Value> &valsForB, bool isA, int a, int b,
                  const LLVMTypeConverter *typeConverter,
                  ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<unsigned> elemsPerThreadIn,
                  ArrayRef<unsigned> elemsPerThread,
                  METAX::IndexCacheInfoSm indexCacheInfoSm,
                  SmallVector<int> numRep, bool enSmIdxCache, bool enSmIndexOpt,
                  CacheMap &cacheMap, bool enableLdsTrans) {
  auto tensorTy = cast<TensorOrMemDesc>(tensor.getType());
  auto ctx = rewriter.getContext();
  // We assumes that the input operand of Dot should be from shared layout.
  // TODO(Superjomn) Consider other layouts if needed later.
  auto sharedLayout = cast<SharedEncodingAttr>(tensorTy.getEncoding());
  // for row maj, order is 1, 0; for col maj, it is 0, 1
  auto order = sharedLayout.getOrder();
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int outVec = sharedLayout.getVec();
  const int inVec = elemsPerThreadIn[order[0]];
  const int minVec = std::min(outVec, inVec);
  const int elemBytes = tensorTy.getElementTypeBitWidth() / 8;

  // the original register_lds2, but discard the prefetch logic.
  auto ld2 = [](ValueTable &vals, int mn, int k, Value val) {
    vals[{mn, k}] = val;
  };

  // the original register_lds2, but discard the prefetch logic.
  auto ld2Opt = [&](ValueTable &vals, int mn, int k,
                    SmallVector<Value> val_vec) {
    for (int j = 0; j < elemsPerThread[order[1]]; ++j) {
      for (int i = 0; i < elemsPerThread[order[0]]; ++i) {
        if (kOrder != order[0]) { // a [0,1], b [1,0]
          vals[{mn * elemsPerThread[order[0]] + i,
                k * elemsPerThread[order[1]] + j}] =
              val_vec[j * elemsPerThread[order[0]] + i];
        } else { // a [1,0], b[0,1]
          vals[{mn * elemsPerThread[order[1]] + j,
                k * elemsPerThread[order[0]] + i}] =
              val_vec[j * elemsPerThread[order[0]] + i];
        }
      }
    }
  };

  // (a, b) is the coordinate. threadShape == matShape in MACA.
  MACAMMASmemLoader loader(wpt, sharedLayout.getOrder(), kOrder,
                           smemObj.strides, tensorTy.getShape() /*tileShape*/,
                           threadShape, perPhase, maxPhase, elemBytes,
                           enSmIdxCache, enSmIndexOpt, rewriter, typeConverter,
                           loc, elemsPerThreadIn, enableLdsTrans);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
  Value smemBase;
  if (op.getIsConstantOffs()) {
    smemBase = smemObj.base;
  } else {
    smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
  }
  Type dataTy = tensorTy.getElementType();
  Type llvmTy = typeConverter->convertType(dataTy);
#ifndef USE_MACA_OPAQUE_PTR
  Type smemPtrTy = ptr_ty(llvmTy, 3);
#else
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);
#endif

  if (elemBytes == 8 || elemBytes == 4 || elemBytes == 2) {
    SmallVector<Value> offs = loader.computeTOffsets(
        warpId, lane, cSwizzleOffset, (kOrder == 1) ? a : b /*mat0*/,
        (kOrder == 1) ? b : a /*mat1*/, numRep, minVec, outVec, tensor,
        rewriter, isA, elemsPerThreadIn, indexCacheInfoSm, cacheMap, mmaLayout,
        loc);

    SmallVector<Value> ptrs(offs.size());
    for (int i = 0; i < offs.size(); ++i) {
      ptrs[i] = bitcast(gep(smemPtrTy, llvmTy, smemBase, ValueRange({offs[i]})),
                        smemPtrTy);
    }

    if (op.getMmaMode() == TT_MODE) {
      valsForB = loader.loadXN(op, ptrs, elemBytes, llvmTy,
                               smemObj.static_offset, mmaLayout);
    } else {
      auto regs = loader.loadXN(op, ptrs, elemBytes, llvmTy,
                                smemObj.static_offset, mmaLayout);
      ld2Opt(vals, a, b, regs);
    }
  } else if (elemBytes == 1) {
    SmallVector<Value> offs = loader.computeTOffsets(
        warpId, lane, cSwizzleOffset, (kOrder == 1) ? a : b /*mat0*/,
        (kOrder == 1) ? b : a /*mat1*/, numRep, minVec, outVec, tensor,
        rewriter, isA, elemsPerThreadIn, indexCacheInfoSm, cacheMap, mmaLayout,
        loc);
    SmallVector<Value> ptrs(offs.size());
    for (int i = 0; i < offs.size(); ++i) {
      ptrs[i] = bitcast(gep(smemPtrTy, llvmTy, smemBase, ValueRange({offs[i]})),
                        smemPtrTy);
    }
    auto regs = loader.loadXN(op, ptrs, elemBytes, llvmTy,
                              smemObj.static_offset, mmaLayout);
    ld2Opt(vals, a, b, regs);
  } else {
    assert(false && "Only support mma MMA_16X16X4F32!");
  }
}

Value loadArg(triton::gpu::LocalLoadOp op, ConversionPatternRewriter &rewriter,
              Location loc, Value tensor, DotOperandEncodingAttr encoding,
              const SharedMemoryObject &smemObj,
              const LLVMTypeConverter *typeConverter, Value thread, bool isA,
              ArrayRef<unsigned> elemsPerThread,
              METAX::IndexCacheInfoSm indexCacheInfoSm, bool enSmIdxCache,
              bool enSmIndexOpt) {
  auto tensorTy = cast<TensorOrMemDesc>(tensor.getType());
  auto sharedLayout = cast<SharedEncodingAttr>(tensorTy.getEncoding());

  SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                             tensorTy.getShape().end());
  auto order = sharedLayout.getOrder();
  auto mmaLayout = cast<MmaEncodingAttr>(encoding.getParent());
  ValueTable vals;
  std::function<void(int, int)> loadFn;
  bool enableLdsTrans = isA ? mmaLayout.getIsATrans() : mmaLayout.getIsBTrans();
  auto mmaThreadShape = getMmaThreadShape(enableLdsTrans);
  // elemsPerThread when load, diff from elemsPerThread on regs when
  // enableLdsTrans.
  auto elemTy = tensorTy.getElementType();
  auto elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  auto elemsPerThreadIn = mmaLayout.getElemsPerThreadOrTrans(encoding);
  // check valid elemsPerThreadIn
  if (enableLdsTrans) {
    auto elemsStride = isA ? mmaLayout.getElementsStride()[0]
                           : mmaLayout.getElementsStride()[1];
    // M/N
    assert(elemsPerThreadIn[order[0]] ==
           elemsPerThread[order[0]] * elemsStride);
    // K
    assert(elemsPerThreadIn[order[1]] ==
           elemsPerThread[order[1]] / elemsStride);
  } else {
    assert(elemsPerThreadIn[0] == elemsPerThread[0]);
    assert(elemsPerThreadIn[1] == elemsPerThread[1]);
  }
  auto mmaThreadM = mmaThreadShape[0];
  auto mmaThreadN = mmaThreadShape[1];
  auto mmaThreadK = mmaThreadShape[2];

  auto wpts = mmaLayout.getWarpsPerCTA();
  int numRepM = 0;
  int numRepN = 0;
  int numRepK = 0;

  if (isA) {
    numRepM = getNumRepM(shape[0], wpts, elemsPerThread[0]);
    numRepK = getNumRepK(shape[1], elemsPerThread[1]);
  } else {
    numRepK = getNumRepK(shape[0], elemsPerThread[0]);
    numRepN = getNumRepN(shape[1], wpts, elemsPerThread[1]);
  }

  SmallVector<int> numRep{numRepM, numRepN, numRepK};

  Value _64 = i32_val(64);
  Value lane = urem(thread, _64);
  Value warp = udiv(thread, _64);

  Value warpM;
  Value warpN;
  int wpt = 0;

  // bsm index cache map: for A: key is {m, k}, for B: key is {k, n}
  CacheMap cacheMap;

  if (isA) {
    bool shared = isa<SharedEncodingAttr>(tensorTy.getEncoding());
    bool block = isa<BlockedEncodingAttr>(tensorTy.getEncoding());

    // tensor -> share memory
    if (isa<SharedEncodingAttr>(tensorTy.getEncoding())) {
      // shape[0] -> M
      warpM = getWarpM(rewriter, loc, shape[0], elemsPerThreadIn[0], wpts, warp,
                       enableLdsTrans);
      // load from smem
      // each warp processes 16*4 elements, according to the instr.
      auto threadShapeM = getMmaThreadShape(enableLdsTrans)[0];
      auto matrixShapeM = threadShapeM * elemsPerThreadIn[0];
      wpt = std::min<int>(wpts[0], shape[0] / matrixShapeM);
    } else if (isa<BlockedEncodingAttr>(tensorTy.getEncoding())) {
      // load from registers, used in gemm fuse
      // TODO(Superjomn) Port the logic.
      assert(false && "Loading A from register is not supported yet.");
    } else {
      assert(false && "A's layout is not supported.");
    }

    SmallVector<Value> valsForB;

    // step1. Perform loading.
    for (int m = 0; m < numRepM; ++m)
      for (int k = 0; k < numRepK; ++k) {
        // kOrder always is 1 for A, no matter it is row maj or col maj
        loadMatrixFn(op, tensor, smemObj, mmaLayout, wpt /*wpt*/, 1 /*kOrder*/,
                     {mmaThreadM, mmaThreadK} /*threadShape*/, warpM /*warpId*/,
                     lane /*laneId*/, vals /*vals*/, valsForB, isA /*isA*/, m,
                     k, typeConverter /* typeConverter */,
                     rewriter /*rewriter*/, loc /*loc*/, elemsPerThreadIn,
                     elemsPerThread, indexCacheInfoSm, numRep, enSmIdxCache,
                     enSmIndexOpt, cacheMap, enableLdsTrans);
      }

    // step2. Format the values to LLVM::Struct to passing to mma codegen.
    return composeValuesToDotOperandLayoutStruct(
        vals, numRepM, numRepK, typeConverter, loc, rewriter, elemsPerThread[0],
        elemsPerThread[1]);
  } else {
    Value warpN = getWarpN(rewriter, loc, shape[1], elemsPerThreadIn[1], wpts,
                           warp, enableLdsTrans);
    SmallVector<Value> valsForB;
    // each warp processes 16*4 elements, according to the Instr.
    auto threadShapeN = getMmaThreadShape(enableLdsTrans)[1];
    auto matrixShapeN = threadShapeN * elemsPerThreadIn[1];
    int wpt = std::min<int>(wpts[1], shape[1] / matrixShapeN);
    for (int n = 0; n < std::max(numRepN, 1); ++n)
      for (int k = 0; k < numRepK; ++k) {
        // kOrder always is 0 for B, no matter it is row maj or col maj
        loadMatrixFn(op, tensor, smemObj, mmaLayout, wpt /*wpt*/, 0 /*kOrder*/,
                     {mmaThreadK, mmaThreadN} /*threadShape*/, warpN /*warpId*/,
                     lane /*laneId*/, vals /*vals*/, valsForB, isA /*isA*/, n,
                     k, typeConverter /* typeConverter */,
                     rewriter /*rewriter*/, loc /*loc*/, elemsPerThreadIn,
                     elemsPerThread, indexCacheInfoSm, numRep, enSmIdxCache,
                     enSmIndexOpt, cacheMap, enableLdsTrans);
      }

    // elemsPerThread: [elemsK, elemsN].
    if (op.getMmaMode() == TT_MODE) {
      return composeValuesToDotOperandLayoutStructForB(
          valsForB, std::max(numRepN, 1), numRepK, typeConverter, loc, rewriter,
          elemsPerThread[1], elemsPerThread[0]);
    } else {
      return composeValuesToDotOperandLayoutStruct(
          vals, std::max(numRepN, 1), numRepK, typeConverter, loc, rewriter,
          elemsPerThread[1], elemsPerThread[0]);
    }
  }
}
} // namespace MACA
} // namespace mlir

using namespace mlir::MACA;

namespace SharedToDotOperandMMAMACA {
Value convertLayout(int opIdx, triton::gpu::LocalLoadOp op,
                    ConversionPatternRewriter &rewriter, Location loc,
                    Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread,
                    ArrayRef<unsigned> elemsPerThread,
                    METAX::IndexCacheInfoSm indexCacheInfoSm, bool enSmIdxCache,
                    bool enSmIndexOpt) {
  if (opIdx == 0)
    return loadArg(op, rewriter, loc, tensor, encoding, smemObj, typeConverter,
                   thread, true, {elemsPerThread[0], elemsPerThread[2]},
                   indexCacheInfoSm, enSmIdxCache, enSmIndexOpt); // m, k
  else {
    assert(opIdx == 1);
    return loadArg(op, rewriter, loc, tensor, encoding, smemObj, typeConverter,
                   thread, false, {elemsPerThread[2], elemsPerThread[1]},
                   indexCacheInfoSm, enSmIdxCache, enSmIndexOpt); // k, n
  }
}
} // namespace SharedToDotOperandMMAMACA
