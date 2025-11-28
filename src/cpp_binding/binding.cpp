/**
 * pybind11 Binding Layer
 * Exposes C++/CUDA functions to Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// Forward declaration from vector_ops.cu
std::vector<int> add_vectors_gpu(const std::vector<int>& a, const std::vector<int>& b);

// Python module definition
PYBIND11_MODULE(planar_cuda, m) {
    m.doc() = "LCN Solver - CUDA Accelerated Backend";

    // Phase 1: Simple vector addition test
    m.def("add_vectors", &add_vectors_gpu, 
          "Add two integer vectors using CUDA",
          py::arg("a"), py::arg("b"));

    // Module metadata
    m.attr("__version__") = "0.1.0-phase1";
    m.attr("cuda_enabled") = true;
}
