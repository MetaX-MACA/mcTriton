add_triton_library(TritonMETAXGPUTransforms
  AccelerateMETAXMatmul.cpp
  MACACommon.cpp
  PipelineMACA.cpp
  PrefetchMACA.cpp
  MoveDotOperandsOutLoop.cpp
  PipelineAsyncTN.cpp
  PipelineAsyncTT.cpp
  PipelineAsyncBase.cpp
  MergeEqualSharedLayout.cpp
  PipelineAsyncMultiDotMLAMixed.cpp
  ChangeLayoutFromRepNToElemN.cpp
  OptimizeCStorePass.cpp
  ChangeLayoutForInt8.cpp
  ChangeConvertLayoutAttr.cpp
  ChangeLayoutForConstancyLoad.cpp
  MergeConvertLayout.cpp

  DEPENDS
  TritonMETAXGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonMETAXGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMETAXGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
