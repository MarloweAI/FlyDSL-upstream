if(WIN32)
  set(_fly_ktrace_object_arg OBJECT)
endif()

add_mlir_dialect_library(MLIRFlyKtraceDialect
  ${_fly_ktrace_object_arg}
  Dialect.cpp

  DEPENDS
  MLIRFlyKtraceIncGen
)
