#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mtf_ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mtf_cpp, m) {
    m.doc() = "C++ backend for MTFLibrary operations";

    m.def("add_mtf_cpp", &add_mtf_cpp, "A function that adds two MTFs using C++");
    m.def("multiply_mtf_cpp", &multiply_mtf_cpp, "A function that multiplies two MTFs using C++");
}
