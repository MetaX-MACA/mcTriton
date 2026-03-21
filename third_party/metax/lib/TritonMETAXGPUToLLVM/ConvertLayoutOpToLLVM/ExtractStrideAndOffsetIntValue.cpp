#include "ExtractStrideAndOffsetIntValue.h"

void printModuleOp(Value value) {
  auto valueOp = value.getDefiningOp();
  auto valueOpBlock = valueOp->getBlock();
  valueOpBlock->printAsOperand(llvm::outs(), true);
  auto terminator = valueOpBlock->getTerminator();
  ModuleOp moduleOp = terminator->getParentOfType<ModuleOp>();
  if (moduleOp) {
    llvm::outs() << "print printModuleOp module.\n";
    moduleOp.print(llvm::outs());
    llvm::outs() << "print printModuleOp module end.\n";
  } else {
    llvm::outs() << "printModuleOp no module.\n";
  }
}

// Support 6 MMA modes, for example:
// 0.test_core.py::test_dot[64-64-64-4-False-False-none-True-float32-float32]
// 1.test_opt_mma.py::test_opt_mma[512-512-512-None-dtype0-False-128-128-32-8-8-2-True-True-True-True]
// 2.test_opt_mma.py::test_opt_mma[1024-1024-512-None-dtype48-False-64-128-64-8-8-1-True-True-True-True]
// 3.test_opt_mma.py::test_opt_mma[512-512-512-None-dtype2-False-128-128-32-8-8-2-True-True-False-True]
// 4.test_matmul.py::test_op[64-32-64-1-2-4-128-64-128-False-False-float16-float16]
// 5.test_matmul.py::test_op[128-64-16-1-4-4-256-128-80-False-False-float16-float16]
int getStrideIntType(Value sStride, ConversionPatternRewriter &rewriter) {
  debug_print << "sStride: " << sStride << "\n";

  // For Debug, open it.
  if (false) {
    printModuleOp(sStride);
  }

  int strideInt = -1;
  // if sStride comes from ExtractValueOp
  if (auto extractOp = sStride.getDefiningOp<LLVM::ExtractValueOp>()) {
    strideInt = findConstant(sStride, -1, rewriter);
  } else {
    debug_print << "sStride is not ExtractValueOp, sStride: " << sStride
                << "\n";
  }
  return strideInt;
}

void getTargetInsertOpForOffset(int targetIndex, LLVM::InsertValueOp insertOp,
                                mlir::Value &subInsertOp) {
  if (targetIndex == insertOp.getPosition()[0]) {
    subInsertOp = insertOp;
  } else {
    while (auto InsertOp2 = subInsertOp.getDefiningOp<LLVM::InsertValueOp>()) {
      auto InsertOp2Pos = InsertOp2.getPosition();
      if (targetIndex == InsertOp2Pos[0]) {
        subInsertOp = InsertOp2;
        break;
      }
      subInsertOp = InsertOp2.getODSOperands(0)[0];
    }
  }
}

// Support 5 MMA modes, for example:
// 0.test_core.py::test_dot[64-64-64-4-False-False-none-True-float32-float32]
// 1.test_opt_mma.py::test_opt_mma[512-512-512-None-dtype0-False-128-128-32-8-8-2-True-True-True-True]
// 2.test_opt_mma.py::test_opt_mma[1024-1024-512-None-dtype48-False-64-128-64-8-8-1-True-True-True-True]
// 3.test_matmul.py::test_op[64-32-64-1-2-4-128-64-128-False-False-float16-float16]
// 4.test_matmul.py::test_op[128-64-16-1-4-4-256-128-80-False-False-float16-float16]
int swizzleOffsetIntType(Value cSwizzleOffset, bool isNotNeedSwizzleOffset,
                         ConversionPatternRewriter &rewriter) {
  debug_print << "cSwizzleOffset is : " << cSwizzleOffset << "\n";

  if (isNotNeedSwizzleOffset) {
    debug_print
        << "swizzleOffsetIntType not parse Offset, isNotNeedSwizzleOffset: "
        << isNotNeedSwizzleOffset << "\n";
    return -1;
  }

  // For Debug, open it.
  if (false) {
    printModuleOp(cSwizzleOffset);
  }

  int offsetInt = -1;
  // if cSwizzleOffset comes from ExtractValueOp
  if (auto extractOp = cSwizzleOffset.getDefiningOp<LLVM::ExtractValueOp>()) {
    offsetInt = findConstant(cSwizzleOffset, -1, rewriter);
  } else {
    debug_print << "cSwizzleOffset is not ExtractValueOp, cSwizzleOffset: "
                << cSwizzleOffset << "\n";
  }
  return offsetInt;
}

/**
 * Input:
 *  value oprand: current oprand
 *  int succExtractValueIdx:
 *      if current oprand's successors contain ExtractValueOp, record the index
 * of ExtractValueOp, index is used and reset in matched InsertValueOp
 */
int findConstant(Value oprand, int succExtractValueIdx,
                 ConversionPatternRewriter &rewriter) {
  // found constantOp, return constant value
  if (auto constantOp = oprand.getDefiningOp<LLVM::ConstantOp>()) {
    debug_print << "Found llvm constantOp:" << constantOp << "\n";
    // return value
    auto constantOpAttr = constantOp.getValue();
    if (auto constantOpInt = dyn_cast<IntegerAttr>(constantOpAttr)) {
      return constantOpInt.getInt();
    } else {
      debug_print << "llvm constantOpAttr is not IntegerAttr,constantOpAttr:"
                  << constantOpAttr << "\n";
      return -1;
    }
  }

  if (auto constantOp = oprand.getDefiningOp<mlir::arith::ConstantOp>()) {
    debug_print << "Found arith constantOp:" << constantOp << "\n";
    // return value
    auto constantOpAttr = constantOp.getValue();
    if (auto constantOpInt = mlir::dyn_cast<IntegerAttr>(constantOpAttr)) {
      return constantOpInt.getInt();
    } else {
      debug_print << "arith constantOpAttr is not IntegerAttr,constantOpAttr:"
                  << constantOpAttr << "\n";
      return -1;
    }
  }

  // if (oprand.isa<BlockArgument>()) {
  if (auto blockArg = dyn_cast<BlockArgument>(oprand)) {
    // if oprand is BlockArgument, find non-loop predecessor block
    // and find the oprand defined in non-loop pred block
    // auto blockArg = oprand.dyn_cast<BlockArgument>();
    Block *curBlock = blockArg.getOwner();
    int arg_index = blockArg.getArgNumber();
    auto succBlocks = curBlock->getSuccessors();
    auto predBlocks = curBlock->getPredecessors();

    Block *nonLoopPreBlock;
    // if predBB is successor of curBlock, then it is a loop : predBB ->
    // curBlock -> predBB
    if (std::distance(predBlocks.begin(), predBlocks.end()) > 2) {
      debug_print << "Don't support more than two predecessors, curBlock:";
      curBlock->printAsOperand(debug_print, true);
      debug_print << "\n";
      return -1;
    }
    // TODO: consider situation: predBB -> curBlock -> otherBB -> predBB ?
    for (auto predBB = predBlocks.begin(); predBB != predBlocks.end();
         predBB++) {
      //(*predBB)->printAsOperand(debug_print, true);
      if (std::find(succBlocks.begin(), succBlocks.end(), *predBB) ==
          succBlocks.end()) {
        nonLoopPreBlock = *predBB;
      }
    }
    if (nonLoopPreBlock == nullptr) {
      debug_print << "can't find nonLoopPreBlock, curBlock:";
      curBlock->printAsOperand(debug_print, true);
      debug_print << "\n";
      return -1;
    }
    auto block_terminator = nonLoopPreBlock->getTerminator();
    auto predOprand = block_terminator->getOperand(arg_index);
    return findConstant(predOprand, succExtractValueIdx, rewriter);
  } else {
    // Find pred definingOp of current oprand
    // support Op:
    // InsertValueOp/ExtractValueOp/AddOp/UnrealizedConversionCastOp/ExtractSliceOp
    if (auto insertValueOp = oprand.getDefiningOp<LLVM::InsertValueOp>()) {
      // find Predecessor oprand, use succExtractValueIdx from corresponding
      // extractValueOP
      if (succExtractValueIdx < 0) {
        debug_print << "can't find succExtractValueIdx, insertValueOp:"
                    << insertValueOp << "\n";
        return -1;
      }
      auto tarInsertValueOpRes = insertValueOp.getODSOperands(0)[0];
      getTargetInsertOpForOffset(succExtractValueIdx, insertValueOp,
                                 tarInsertValueOpRes);
      auto tarInsertValueOp =
          tarInsertValueOpRes.getDefiningOp<LLVM::InsertValueOp>();
      if (tarInsertValueOp == nullptr) {
        debug_print << "can't find target InsertValueOp, InsertValueOp:"
                    << insertValueOp
                    << " \n succExtractValueIdx:" << succExtractValueIdx
                    << "\n";
        return -1;
      }
      auto predOprand = tarInsertValueOp.getODSOperands(1)[0];
      // reset succExtractValueIdx
      return findConstant(predOprand, -1, rewriter);

    } else if (auto extractValueOp =
                   oprand.getDefiningOp<LLVM::ExtractValueOp>()) {
      // find Predecessor oprand, and set index
      int extractOpPos = extractValueOp.getPosition()[0];
      auto predOprand = extractValueOp.getODSOperands(0)[0];
      return findConstant(predOprand, extractOpPos, rewriter);

    } else if (auto addOp = oprand.getDefiningOp<LLVM::AddOp>()) {
      auto addOpLhs = addOp.getLhs();
      auto addOpRhs = addOp.getRhs();

      // succExtractValueIdx==-1, shouldn't exist non-matched ExtractValueOp
      if (succExtractValueIdx != -1) {
        debug_print << "Not all ExtractValueOp are matched by InsertValueOp "
                       "before this addOp:"
                    << addOp << "\n";
        return -1;
      }

      // find constant of LHS and RHS
      int addOpLhsConst = findConstant(addOpLhs, succExtractValueIdx, rewriter);
      int addOpRhsConst = findConstant(addOpRhs, succExtractValueIdx, rewriter);
      if (addOpLhsConst == -1 || addOpRhsConst == -1) {
        debug_print << "can't find constant in addOp oprands:" << addOp
                    << " \n addOpLhsConst:" << addOpLhsConst
                    << " \n addOpRhsConst:" << addOpRhsConst << "\n";
        return -1;
      }

      int addOpConst = addOpLhsConst + addOpRhsConst;
      return addOpConst;

    } else if (auto unrealizedConversionCastOp =
                   oprand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      auto predOprand = unrealizedConversionCastOp.getODSOperands(0)[0];
      return findConstant(predOprand, succExtractValueIdx, rewriter);

      // } else if (auto extractSliceOp =
      // oprand.getDefiningOp<triton::gpu::ExtractSliceOp>()) {
    } else if (auto extractSliceOp =
                   oprand.getDefiningOp<tensor::ExtractSliceOp>()) {
      // first mapped to llvmOp
      auto MappedExtractSliceOp = rewriter.getRemappedValue(extractSliceOp);
      return findConstant(MappedExtractSliceOp, succExtractValueIdx, rewriter);
    } else if (auto memDescSubviewOp =
                   oprand.getDefiningOp<triton::gpu::MemDescSubviewOp>()) {
      auto MappedMemDescOp = rewriter.getRemappedValue(memDescSubviewOp);
      return findConstant(MappedMemDescOp, succExtractValueIdx, rewriter);
    } else {
      // Op pattern Not matched, return -1
      debug_print << "Op pattern Not matched, oprand:" << oprand << "\n";
      return -1;
    }
  }
  return -1;
}
