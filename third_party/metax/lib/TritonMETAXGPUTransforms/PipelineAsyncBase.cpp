#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#ifdef USE_MACA
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#endif

using namespace mlir;
namespace ttg = triton::gpu;
namespace tt = mlir::triton;

namespace {

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

struct TritonMETAXGPUPipelineAsyncBasePass
    : public TritonMETAXGPUPipelineAsyncNTBase<
          TritonMETAXGPUPipelineAsyncBasePass> {
  TritonMETAXGPUPipelineAsyncBasePass() = default;
  TritonMETAXGPUPipelineAsyncBasePass(int numStages, bool isFullStage,
                                      bool mixed) {
    this->numStages = numStages;
    this->isFullStage = isFullStage;
    this->mixed = mixed;
  }

  struct AsyncLoadOps {
    Operation *copyView = nullptr;
    Operation *copyIndex = nullptr;
    Operation *copy = nullptr;
    Operation *loadView = nullptr;
    Operation *loadIndex = nullptr;
    Operation *localLoad = nullptr;
    int innerStage = 0;
    bool isA = false;
    unsigned gvmCount = 0;
  };

  static void collectDepsInLoop(Operation *op, scf::ForOp loop,
                                llvm::SmallPtrSetImpl<Operation *> &deps) {
    if (!op || op->getBlock() != loop.getBody() || !deps.insert(op).second)
      return;
    for (Value operand : op->getOperands())
      collectDepsInLoop(operand.getDefiningOp(), loop, deps);
  }

  static void collectAsyncLoadStages(
      Operation *op, scf::ForOp loop,
      const DenseMap<Operation *, int> &localLoadStages,
      llvm::SmallPtrSetImpl<Operation *> &visited, int &readyStage) {
    if (!op || op->getBlock() != loop.getBody() || !visited.insert(op).second)
      return;
    auto it = localLoadStages.find(op);
    if (it != localLoadStages.end()) {
      readyStage = std::max(readyStage, it->second);
      return;
    }
    for (Value operand : op->getOperands())
      collectAsyncLoadStages(operand.getDefiningOp(), loop, localLoadStages,
                             visited, readyStage);
  }

  static Operation *
  findDotOperandConversion(Value value,
                           SmallVectorImpl<Operation *> &conversionChain,
                           RankedTensorType &resultType) {
    while (value.hasOneUse()) {
      Operation *user = *value.getUsers().begin();
      if (isa<ttg::ConvertLayoutOp, ttg::LocalLoadOp>(user)) {
        auto type = dyn_cast<RankedTensorType>(user->getResult(0).getType());
        if (type && isa<ttg::DotOperandEncodingAttr>(type.getEncoding()))
          resultType = type;
        return resultType ? user : nullptr;
      }
      if (!isa<ttg::LocalAllocOp, tt::TransOp>(user) ||
          user->getNumResults() != 1)
        return nullptr;
      conversionChain.push_back(user);
      value = user->getResult(0);
    }
    return nullptr;
  }

  static Value createBufferIndex(OpBuilder &builder, scf::ForOp loop,
                                 int numBuffers) {
    Location loc = loop.getLoc();
    Value offset = arith::SubIOp::create(
        builder, loc, loop.getInductionVar(), loop.getLowerBound());
    Value iteration =
        arith::DivSIOp::create(builder, loc, offset, loop.getStep());
    Type indexType = loop.getInductionVar().getType();
    Value bufferCount = arith::ConstantOp::create(
        builder, loc, indexType,
        builder.getIntegerAttr(indexType, numBuffers));
    return arith::RemUIOp::create(builder, loc, iteration, bufferCount);
  }

  LogicalResult pipelineLoop(scf::ForOp loop,
                             tt::ModuleAxisInfoAnalysis &axisInfo) {
    bool useInnerPrefetch = numStages == 1;
    SmallVector<tt::LoadOp> loads;
    loop.walk([&](tt::LoadOp load) {
      if (load->hasAttr("metax.split_tensor_map.operand") &&
          load->getParentOfType<scf::ForOp>() == loop)
        loads.push_back(load);
    });
    if (loads.empty())
      return failure();

    SmallVector<AsyncLoadOps> loweredLoads;
    for (tt::LoadOp load : loads) {
      if (!tt::canBeAsyncLoad(load))
        return failure();

      auto loadType = cast<RankedTensorType>(load.getType());
      SmallVector<Operation *> dotConversionChain;
      RankedTensorType dotOperandType;
      Operation *dotOperandConversion = findDotOperandConversion(
          load.getResult(), dotConversionChain, dotOperandType);
      RankedTensorType localLoadType =
          dotOperandConversion ? dotOperandType : loadType;
      auto sharedEncoding = tt::getSharedEncoding(load);
      unsigned bufferCount = useInnerPrefetch ? 1 : numStages;
      Value alloc = tt::createAlloc(loop, loadType, load.getLoc(),
                                    sharedEncoding,
                                    /*distance=*/bufferCount);

      OpBuilder builder(load);
      AsyncLoadOps ops;
      ops.innerStage =
          load->getAttrOfType<IntegerAttr>(
                  "metax.split_tensor_map.inner_stage")
              .getInt();
      ops.isA =
          load->getAttrOfType<StringAttr>("metax.split_tensor_map.operand")
              .getValue() == "A";
      ops.gvmCount = getGVMNumberPerOp(load);

      Value copyView;
      if (useInnerPrefetch) {
        copyView = tt::createSingleBufferView(builder, alloc, 0);
        ops.copyIndex = copyView.getDefiningOp()->getOperand(1).getDefiningOp();
      } else {
        Value copyBufferIndex =
            createBufferIndex(builder, loop, numStages);
        ops.copyIndex = copyBufferIndex.getDefiningOp();
        copyView =
            tt::createSingleBufferView(builder, alloc, copyBufferIndex);
      }
      ops.copyView = copyView.getDefiningOp();
      unsigned contiguity = axisInfo.getContiguity(load.getPtr());
      auto copy = ttg::AsyncCopyGlobalToLocalOp::create(
          builder, load.getLoc(), load.getPtr(), copyView, load.getMask(),
          load.getOther(), load.getCache(), load.getEvict(),
          load.getIsVolatile(), /*intrinsic=*/true,
          static_cast<int32_t>(contiguity));
      ops.copy = copy;

      Value loadView;
      if (useInnerPrefetch) {
        loadView = tt::createSingleBufferView(builder, alloc, 0);
        ops.loadIndex = loadView.getDefiningOp()->getOperand(1).getDefiningOp();
      } else {
        Value loadBufferIndex =
            createBufferIndex(builder, loop, numStages);
        ops.loadIndex = loadBufferIndex.getDefiningOp();
        loadView =
            tt::createSingleBufferView(builder, alloc, loadBufferIndex);
      }
      ops.loadView = loadView.getDefiningOp();
      auto localLoad = ttg::LocalLoadOp::create(
          builder, load.getLoc(), localLoadType, loadView);
      ops.localLoad = localLoad;

      if (dotOperandConversion) {
        dotOperandConversion->getResult(0).replaceAllUsesWith(
            localLoad.getResult());
        dotOperandConversion->erase();
        for (Operation *op : llvm::reverse(dotConversionChain)) {
          if (op->use_empty())
            op->erase();
        }
      } else {
        load.getResult().replaceAllUsesWith(localLoad.getResult());
      }
      if (!load.getResult().use_empty())
        return failure();
      load.erase();
      loweredLoads.push_back(ops);
    }

    llvm::sort(loweredLoads, [](const AsyncLoadOps &lhs,
                                const AsyncLoadOps &rhs) {
      if (lhs.innerStage != rhs.innerStage)
        return lhs.innerStage < rhs.innerStage;
      return lhs.isA && !rhs.isA;
    });

    unsigned totalGVM = 0;
    for (const AsyncLoadOps &ops : loweredLoads)
      totalGVM += ops.gvmCount;

    SmallVector<int> innerStages;
    DenseMap<Operation *, int> localLoadStages;
    for (const AsyncLoadOps &ops : loweredLoads) {
      localLoadStages[ops.localLoad] = ops.innerStage;
      if (innerStages.empty() || innerStages.back() != ops.innerStage)
        innerStages.push_back(ops.innerStage);
    }

    DenseMap<int, SmallVector<Operation *, 4>> dotsByReadyStage;
    unsigned numDots = 0;
    unsigned numScheduledDots = 0;
    loop.walk([&](tt::DotOp dot) {
      if (dot->getParentOfType<scf::ForOp>() != loop)
        return;
      ++numDots;
      int readyStage = -1;
      llvm::SmallPtrSet<Operation *, 8> visited;
      for (Value operand : dot->getOperands().take_front(2)) {
        collectAsyncLoadStages(operand.getDefiningOp(), loop, localLoadStages,
                               visited, readyStage);
      }
      if (readyStage >= 0) {
        dotsByReadyStage[readyStage].push_back(dot);
        ++numScheduledDots;
      }
    });
    if (numDots == 0 || numScheduledDots != numDots)
      return failure();

    DenseMap<int, SmallVector<Operation *, 2>> syncOps;
    OpBuilder syncBuilder(loop.getBody()->getTerminator());
    if (useInnerPrefetch) {
      unsigned remainingGVM = totalGVM / 2;
      for (int innerStage : innerStages) {
        auto arrive = syncBuilder.create<ttg::GVMArriveOp>(
            loop.getLoc(), static_cast<int32_t>(remainingGVM));
        auto barrier =
            syncBuilder.create<ttg::BarrierSharedOp>(loop.getLoc());
        syncOps[innerStage] = {arrive, barrier};
      }
    } else {
      DenseMap<int, unsigned> gvmByInnerStage;
      for (const AsyncLoadOps &ops : loweredLoads)
        gvmByInnerStage[ops.innerStage] += ops.gvmCount;
      unsigned currentIterationGVM = totalGVM;
      unsigned futureIterationGVM = (numStages - 2) * totalGVM;
      for (int innerStage : innerStages) {
        currentIterationGVM -= gvmByInnerStage[innerStage];
        unsigned remainingGVM =
            futureIterationGVM + currentIterationGVM;
        auto arrive = syncBuilder.create<ttg::GVMArriveOp>(
            loop.getLoc(), static_cast<int32_t>(remainingGVM));
        auto barrier =
            syncBuilder.create<ttg::BarrierSharedOp>(loop.getLoc());
        syncOps[innerStage] = {arrive, barrier};
      }
    }

    std::vector<std::pair<Operation *, unsigned>> schedule;
    llvm::SmallPtrSet<Operation *, 32> scheduled;
    llvm::SmallPtrSet<Operation *, 32> allCopyDeps;
    llvm::SmallPtrSet<Operation *, 32> preCopyStage;
    for (const AsyncLoadOps &ops : loweredLoads)
      collectDepsInLoop(ops.copy, loop, allCopyDeps);

    // A stage-0 copy consumes loop-carried pointers from the next iteration.
    // Schedule the yield-side pointer increments before that copy.
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (Operation *op : allCopyDeps) {
      for (Value operand : op->getOperands()) {
        auto arg = dyn_cast<BlockArgument>(operand);
        if (!arg || arg.getOwner() != loop.getBody() ||
            arg.getArgNumber() == 0)
          continue;
        Value nextValue = yield.getOperand(arg.getArgNumber() - 1);
        collectDepsInLoop(nextValue.getDefiningOp(), loop, preCopyStage);
      }
    }

    for (Operation &op : loop.getBody()->without_terminator()) {
      if (preCopyStage.contains(&op) && scheduled.insert(&op).second)
        schedule.push_back({&op, 1});
    }

    auto scheduleDeps = [&](Operation *root, unsigned stage) {
      llvm::SmallPtrSet<Operation *, 32> deps;
      collectDepsInLoop(root, loop, deps);
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (deps.contains(&op) && scheduled.insert(&op).second)
          schedule.push_back({&op, stage});
      }
    };

    auto scheduleCopies = [&](int innerStage) {
      for (const AsyncLoadOps &ops : loweredLoads) {
        if (ops.innerStage == innerStage)
          scheduleDeps(ops.copy, 0);
      }
    };

    if (useInnerPrefetch) {
      llvm::SmallPtrSet<Operation *, 8> deferredRegisterPrefetchOps;
      auto scheduleLoads = [&](int innerStage, unsigned pipelineStage) {
        for (Operation *op : syncOps[innerStage]) {
          if (scheduled.insert(op).second)
            schedule.push_back({op, pipelineStage});
        }
        for (const AsyncLoadOps &ops : loweredLoads) {
          if (ops.innerStage != innerStage)
            continue;
          for (Operation *op :
               {ops.loadIndex, ops.loadView, ops.localLoad}) {
            if (deferredRegisterPrefetchOps.erase(op) ||
                scheduled.insert(op).second)
              schedule.push_back({op, pipelineStage});
          }
        }
      };

      auto scheduleDots = [&](int innerStage) {
        for (Operation *dot : dotsByReadyStage[innerStage])
          scheduleDeps(dot, 1);
      };

      // Prefetch the first two A/B register tiles one K iteration ahead. Their
      // local-load results cross pipeline stages and become loop-carried.
      size_t numRegisterPrefetchStages =
          std::min<size_t>(2, innerStages.size());
      for (size_t i = 0; i < numRegisterPrefetchStages; ++i)
        scheduleCopies(innerStages[i]);
      for (const AsyncLoadOps &ops : loweredLoads) {
        if (std::find(innerStages.begin(),
                      innerStages.begin() + numRegisterPrefetchStages,
                      ops.innerStage) ==
            innerStages.begin() + numRegisterPrefetchStages)
          continue;
        for (Operation *op : {ops.loadIndex, ops.loadView, ops.localLoad}) {
          scheduled.insert(op);
          deferredRegisterPrefetchOps.insert(op);
        }
      }
      for (size_t i = 0; i < numRegisterPrefetchStages; ++i)
        scheduleDots(innerStages[i]);

      // Consume and refill the remaining shared slices in order.
      for (size_t i = numRegisterPrefetchStages; i < innerStages.size(); ++i) {
        scheduleLoads(innerStages[i], 1);
        if (i > numRegisterPrefetchStages)
          scheduleCopies(innerStages[i - 1]);
        scheduleDots(innerStages[i]);
      }

      // Refill the last slice after the wraparound synchronization, then
      // prepare the register-prefetched values for the next K iteration.
      if (numRegisterPrefetchStages != 0) {
        scheduleLoads(innerStages[0], 0);
        scheduleCopies(innerStages.back());
        for (size_t i = 1; i < numRegisterPrefetchStages; ++i)
          scheduleLoads(innerStages[i], 0);
      }

      for (Operation &op : loop.getBody()->without_terminator()) {
        if (scheduled.insert(&op).second)
          schedule.push_back({&op, 1});
      }
    } else {
      unsigned loadStage = numStages - 1;
      unsigned dotStage = numStages;
      auto scheduleLoads = [&](int innerStage) {
        for (Operation *op : syncOps[innerStage]) {
          if (scheduled.insert(op).second)
            schedule.push_back({op, loadStage});
        }
        for (const AsyncLoadOps &ops : loweredLoads) {
          if (ops.innerStage != innerStage)
            continue;
          llvm::SmallPtrSet<Operation *, 16> deps;
          collectDepsInLoop(ops.localLoad, loop, deps);
          for (Operation &op : loop.getBody()->without_terminator()) {
            if (deps.contains(&op) && scheduled.insert(&op).second)
              schedule.push_back({&op, loadStage});
          }
        }
      };

      auto scheduleDots = [&](int innerStage) {
        for (Operation *dot : dotsByReadyStage[innerStage])
          scheduleDeps(dot, dotStage);
      };

      // Load the next K tile into registers before reusing the oldest shared
      // slot. Stage 0 issues a more distant async copy while the final stage
      // computes from the loop-carried register tile.
      for (int innerStage : innerStages)
        scheduleLoads(innerStage);
      for (int innerStage : innerStages)
        scheduleCopies(innerStage);
      for (int innerStage : innerStages)
        scheduleDots(innerStage);

      for (Operation &op : loop.getBody()->without_terminator()) {
        if (scheduled.insert(&op).second)
          schedule.push_back({&op, dotStage});
      }
    }

    tt::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = false;
    options.predicateFn = [](RewriterBase &rewriter, Operation *op,
                             Value pred) {
      if (isa<ttg::GVMArriveOp, ttg::BarrierSharedOp>(op))
        return op;
      return tt::wrapInMaskOp(rewriter, op, pred);
    };
    options.getScheduleFn =
        [schedule](scf::ForOp,
                   std::vector<std::pair<Operation *, unsigned>> &result) {
          result = schedule;
        };

    IRRewriter rewriter(loop);
    if (failed(tt::pipelineForLoop(rewriter, loop, options)))
      return failure();
    return success();
  }

  void runOnOperation() override {
    if (numStages < 1)
      return;

    ModuleOp module = getOperation();
    tt::ModuleAxisInfoAnalysis axisInfo(module);
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) {
      if (!loop->hasAttr("metax.split_tensor_map"))
        return;
      if (numStages == 1) {
        auto stageM =
            loop->getAttrOfType<IntegerAttr>("metax.split_tensor_map.stage_m");
        auto stageN =
            loop->getAttrOfType<IntegerAttr>("metax.split_tensor_map.stage_n");
        if (!stageM || !stageN ||
            (stageM.getInt() <= 1 && stageN.getInt() <= 1))
          return;
      }
      loops.push_back(loop);
    });

    bool transformed = false;
    for (scf::ForOp loop : loops) {
      if (failed(pipelineLoop(loop, axisInfo))) {
        loop.emitError("failed to expand the MetaX async pipeline");
        signalPassFailure();
        return;
      }
      transformed = true;
    }
    if (!transformed)
      return;

    tt::resolveMaskOp(module);
    tt::removePipeliningAttributes(module);
  }
};
} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUPipelineAsyncBasePass(int numStages, bool isFullStage,
                                                bool mixed) {
  return std::make_unique<TritonMETAXGPUPipelineAsyncBasePass>(
      numStages, isFullStage, mixed);
}
