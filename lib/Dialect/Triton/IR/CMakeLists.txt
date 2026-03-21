add_triton_library(TritonIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
