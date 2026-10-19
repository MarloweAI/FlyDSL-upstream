add_mlir_dialect_library(MLIRFlyKtraceDialect
  # Windows compiles obj.* into FlyPythonCAPI; Visual Studio makes it only on request.
  OBJECT
  Dialect.cpp

  DEPENDS
  MLIRFlyKtraceIncGen
)
