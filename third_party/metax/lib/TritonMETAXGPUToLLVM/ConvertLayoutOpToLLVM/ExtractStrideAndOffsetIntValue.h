#ifndef EXTRACT_STRIDE_AND_OFFSET_INT_VALUE_H
#define EXTRACT_STRIDE_AND_OFFSET_INT_VALUE_H

#include "../MACACommonConversion.h"

void printModuleOp(Value value);

// from value type to int type
int getStrideIntType(Value sStride, ConversionPatternRewriter &rewriter);

void getTargetInsertOpForOffset(int targetIndex, LLVM::InsertValueOp insertOp,
                                mlir::Value &subInsertOp);

// from value type to int type
int swizzleOffsetIntType(Value cSwizzleOffset, bool isNotNeedSwizzleOffset,
                         ConversionPatternRewriter &rewriter);

int findConstant(Value oprand, int succExtractValueIdx,
                 ConversionPatternRewriter &rewriter);

#endif
