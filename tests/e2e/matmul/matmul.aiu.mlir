
!A_size = tensor<3x5xf32>
!B_size = tensor<5x3xf32>
!C_size = tensor<3x3xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}

// RUN: iree-compile %s --iree-hal-target-backends=aiunite --iree-opt-data-tiling=false --iree-codegen-use-transform-dialect-strategy=%p/matmul_codegen_default_spec.mlir

// EXEC: 3x3xf32=[52 52 52][52 52 52][52 52 52]
