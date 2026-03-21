add_triton_library(TritonMETAXGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAMACA.cpp
    ConvertLayoutOpToLLVM/ExtractStrideAndOffsetIntValue.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/MMAMACA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    ViewOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp

    DEPENDS
    TritonMETAXGPUTransformsIncGen
    TritonMETAXGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)
