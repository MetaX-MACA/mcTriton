#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "convert layout";

  //USE_MACA
  let arguments = (ins TT_Tensor:$src,
                   DefaultValuedAttr<BoolAttr, "false">:$intrinsic,
                   DefaultValuedAttr<BoolAttr, "true">:$withPad);

  //let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//USE_MACA
def TTG_GVMArriveOp: TTG_Op<"gvm_arrive"> {
  let summary = "global memory arrive";

  let arguments = (ins I32Attr:$num);

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

//USE_MACA
def TTG_SchedBoundOp: TTG_Op<"sched_bound"> {
  let summary = "set up schedule bound";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

//USE_MACA
def TTG_IGLPOp: TTG_Op<"iglp"> {
  let summary = "";

  let arguments = (ins DefaultValuedAttr<I32Attr, "0">:$config_0,
                   DefaultValuedAttr<I32Attr, "-1">:$config_1,
                   DefaultValuedAttr<I32Attr, "-1">:$config_2,
                   DefaultValuedAttr<I32Attr, "-1">:$config_3,
                   DefaultValuedAttr<I32Attr, "-1">:$config_4,
                   DefaultValuedAttr<I32Attr, "-1">:$config_5,
                   DefaultValuedAttr<I32Attr, "-1">:$config_6,
                   DefaultValuedAttr<I32Attr, "-1">:$config_7);

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

//USE_MACA
def TTG_BarrierOp: TTG_Op<"barrier"> {
  let summary = "set up intrinsic barrier";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

//USE_MACA
def TTG_BarrierSharedOp: TTG_Op<"barrier_shared"> {
  let summary = "set up intrinsic barrier_shared";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

//USE_MACA
def TTG_ExtractTensorOp: TTG_Op<"extract_tensor", [Pure,
                                                   DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "extract tensor operation";
  let description = [{
    pipeline_async_tn needs to segment the A, B, and C matrices along the m and n directions,
    and needs a triton op to represent the semantics of record segmentation.
    extract_tensor [ctaIdx] [elemIdx]: tensor ptr/data -> subtensor ptr/data,
    ctaIdx and elemIdx are variable length arrays, indicate the data idx to be extracted,
    ctaIdx is linearized CTA index, and elemIdx is linearized elem index.
    The shape size and sizePerThread of layout can be changed.
    The extract of ptr only supports blockedlayout. Data extract supports mma.
    For example:
      the totalElemsPerThread layout be like:
      --CTA0-------CTA1--                    ----CTA0---------CTA1----
      0 1 2 3    0 1 2 3                     0  1  2  3    8  9  10 11
      4 5 6 7    4 5 6 7                     4  5  6  7    12 13 14 15
      --CTA2-------CTA3--   ===linearIdx===> ----CTA2---------CTA3----
      0 1 2 3    0 1 2 3                     16 17 18 19   24 25 26 27
      4 5 6 7    4 5 6 7                     20 21 22 23   28 29 30 31
      sizePerThread = [2, 4]
      totalElemsPerThread = [4, 8]
      order = [1, 0]
      numOfReplica = [2, 2]

      when ctaIdx = [0, 1] and elemIdx = [0, 1, 2, 3, 4, 5, 6, 7],
      the extracted data are:
      ----CTA0---------CTA1----
      0  1  2  3    8  9  10 11
      4  5  6  7    12 13 14 15

      when ctaIdx = [0, 1, 2, 3] and elemIdx = [0, 1, 2, 3],
      the extracted data are:
      ----CTA0---------CTA1----
      0  1  2  3    8  9  10 11
      ----CTA2---------CTA3----
      16 17 18 19   24 25 26 27
  }];

  let arguments = (ins
    AnyTypeOf<[TT_TensorPtr, TT_Tensor]>:$source,
    DenseI64ArrayAttr:$ctaIdx,
    DenseI64ArrayAttr:$elemIdx
  );
  let results = (outs AnyTypeOf<[TT_TensorPtr, TT_Tensor]>:$result); 
}

//USE_MACA
def TTG_InsertTensorOp: TTG_Op<"insert_tensor", [Pure,
                                                 DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "extract tensor operation";
  let description = [{
    pipeline_async_tn needs to segment the A, B, and C matrices along the m and n directions,
    and needs a triton op to represent the semantics of record segmentation.
    insert_tensor [ctaIdx] [elemIdx]:  subtensor data into tensor data,
    ctaIdx and elemIdx are variable length arrays, indicate the data idx to be extracted,
    ctaIdx is linearized CTA index, and elemIdx is linearized elem index.    
    insert the register in subtensor into the struct of the original tensor to ensure the continuity of ir.
    Like extract_tensor, shape size and sizePerThread of layout can be changed.
  }];

  let arguments = (ins
    TT_Tensor:$inserted,
    TT_Tensor:$insert,
    DenseI64ArrayAttr:$ctaIdx,
    DenseI64ArrayAttr:$elemIdx
  );
  let results = (outs TT_Tensor:$result);
}

//USE_MACA
def TTG_SwizzleTensorOp: TTG_Op<"swizzle_tensor", [Pure,
                                                   SameOperandsAndResultEncoding,
                                                   DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "swizzle tensor operation";
  let description = [{
    The principle of swizzling in this op is the same as SharedEncodingAttr, so as the arguments of
    this op. But instead of exchanging addresses, the tensor values are recalculated according to the swizzeling order.
    Notice: only the first element of vector will be recalculated currently, other elements of the
    vector will remain the same.
  }];

  let arguments = (ins
    TT_Tensor:$src,
    I32Attr:$InVec,
    I32Attr:$OutVec,
    I32Attr:$perPhase,
    I32Attr:$maxPhase
  );
  let results = (outs TT_Tensor:$result);
}

def TTG_BsmPermOp: TTG_Op<"bsm_perm",
                          [SameOperandsAndResultShape,
                           Pure]> {
  let summary = "separate perm and lds.";
  let description = [{
    separate perm and lds.
  }];

  let arguments = (ins
    TT_Tensor:$src1
  );
  let results = (outs TT_Tensor:$result);
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins Variadic<TTG_AsyncToken>:$asyncToken, I32Attr:$num);

  let results = (outs TTG_AsyncToken:$retToken);

  let assemblyFormat = "$asyncToken attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncCommitGroupOp : TTG_Op<"async_commit_group"> {
  let summary = "async commit group";

  let results = (outs TTG_AsyncToken:$asyncToken);
  let arguments = (ins Variadic<TTG_AsyncToken>:$inputTokens);

  let assemblyFormat = [{
    $inputTokens attr-dict
  }];

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncCopyGlobalToLocalOp : TTG_Op<"async_copy_global_to_local", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"infer mask type from src type",
                 "src", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
  TypesMatchWith<"infer other type from src type",
                 "src", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 4) || std::equal_to<>()">
]> {
  let summary = "copy data from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory asynchronously.
    This is analogue to tt.load except the data are copied to local memory pointed
    by by the memory descriptor instread of a distributed tensor. The rest of the
    operands are the same as tt.load.
  }];

  let arguments = (
    ins TT_PtrTensor:$src,
    TT_MemDescType:$result,
    Optional<I1Tensor>:$mask,
    Optional<TT_Type>:$other,
    DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile,
    DefaultValuedAttr<BoolAttr, "false">:$intrinsic
  );

  let builders = [
      OpBuilder<(ins "Value":$src, "Value":$result,
                     "triton::CacheModifier":$cache,
                     "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
  ];

  let results = (outs TTG_AsyncToken:$token);

  let extraClassDeclaration = [{
    static DenseSet<unsigned> getEligibleLoadByteWidth(int computeCapability) {
      DenseSet<unsigned> validLoadBytes;
      if (computeCapability >= 80) {
        validLoadBytes = {2, 4, 8, 16};
      }
      return validLoadBytes;
    }
  }];

  // Specify cacheModifier and evictionPolicy explicitly, instead of leaving
  // them in attr-dict, because this way their values get printed as strings,
  // rather than as opaque integers.
  //
  // Note there are no commas between other, cacheModifier, and evictionPolicy,
  // due to limitations in MLIR's asm parser.
  let assemblyFormat = [{
    $src `,` $result (`mask` $mask^)? (`other` $other^)?
    oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
    attr-dict `:` type($src) `->` type($result)
  }];
}


// Allocate shared memory
def TTG_LocalAllocOp : TTG_Op<"local_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate tensor";
  let description = [{
    This operation allocates buffer in shared memory and return a descriptor
    containing the address and a view of the buffer.

    Explicitly deallocating a buffer is optional; see local_dealloc.
  }];
  let arguments = (ins Optional<TT_Tensor>:$src);

  let assemblyFormat = [{$src attr-dict `:` functional-type(operands, results)}];

  let results = (outs TT_MemDescType:$result);
}

// Deallocate shared memory
def TTG_LocalDeallocOp : TTG_Op<"local_dealloc", [MemoryEffects<[MemFree<SharedMemory>]>]> {
  let summary = "dealloc buffer";

  let description = [{
    This operation deallocates a buffer explicitly. Using the buffer after this
    operation is undefined.

    This operation is optional.  If you don't explicitly dealloc a buffer, the
    compiler assumes it's deallocated at the first point that post-dominates all
    uses of the alloc.

    Because we assume a memdesc is dead at the first point that post-dominates
    its uses, ops that wait for an async operation on a memdesc to complete
    (such as triton_nvidia_gpu.dot_wait) should also take the memdesc as an
    operand.
  }];

  let arguments = (ins TT_MemDescType:$src);

  // Use qualified() otherwise "!tt.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src attr-dict `:` qualified(type($src))}];
}

def TTG_MemDescSubviewOp : TTG_Op<"memdesc_subview", [Pure]> {
  let summary = "take a subview of the descriptor.";

  let description = [{
    This operation returns a new descriptor representing a subview of the buffer.
    It doesn't affect the underlying memory. The subview can be rank-reduced.

    For example, suppose that
     - the input shape is 2x4x16xf16,
     - the output shape is 4x4xf16, and
     - offsets = [1, 0, 4].

    Then in Python syntax, the subview covers input[1][0:4][4:8].
  }];
  let arguments = (
    ins TT_MemDescType:$src, Variadic<I32>:$offsets,
    DefaultValuedAttr<BoolAttr, "false">:$isConstantOffs);

  // Use qualified() otherwise "!tt.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src `[` $offsets `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))}];

  let results = (outs TT_MemDescType:$result);

  let hasVerifier = 1;
}

def TTG_LocalLoadOp : TTG_Op<"local_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load a buffer from local memory into a distributed tensor";

  let description = [{
    Load a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (ins TT_MemDescType:$src, Optional<TTG_AsyncToken> :$token, 
                   DefaultValuedAttr<BoolAttr, "false">:$intrinsic,
                   DefaultValuedAttr<BoolAttr, "false">:$isConstantOffs,
                   DefaultValuedAttr<I32Attr, "-1">:$mmaMode);

  let builders = [
      OpBuilder<(ins "Type":$retType, "Value":$src),
      [{
      build($_builder, $_state, retType, src, /*token=*/static_cast<mlir::Value>(nullptr));
      }]>];

  // Use qualified() otherwise "!tt.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{$src (`token` $token^)? attr-dict `:` qualified(type($src)) `->` type($result)}];

  let results = (outs TT_Tensor:$result);
}

def TTG_LocalStoreOp : TTG_Op<"local_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store a distributed tensor into a buffer in local memory";

  let description = [{
    Store a distributed tensor into a buffer in local memory.
  }];
  let arguments = (ins TT_Tensor:$src, TT_MemDescType:$dst,
                   DefaultValuedAttr<BoolAttr, "false">:$intrinsic);

  // Use qualified() otherwise "!tt.memdesc<X>" is printed as "<X>".
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `->` qualified(type($dst))
  }];
}

#endif
