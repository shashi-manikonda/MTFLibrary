#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mtf_data.hpp"
namespace py = pybind11;

PYBIND11_MODULE(mtf_cpp, m) {
    m.doc() = "C++ backend for MTFLibrary operations";

    py::class_<MtfData>(m, "MtfData")
        .def(py::init<>())
        .def(py::init<int>())
        .def("to_dict", &MtfData::to_dict, "Converts the MtfData object to a dictionary of numpy arrays.")
        .def("from_numpy", &MtfData::from_numpy, "Initializes MtfData from numpy arrays.")
        .def("add", &MtfData::add, "Adds another MtfData object.")
        .def("add_inplace", &MtfData::add_inplace, "Adds another MtfData object in-place.")
        .def("multiply", &MtfData::multiply, "Multiplies by another MtfData object.")
        .def("multiply_inplace", &MtfData::multiply_inplace, "Multiplies by another MtfData object in-place.")
        .def("negate", &MtfData::negate, "Negates the MtfData object.");

    m.def("switch_backend", [](const std::string& backend_name) {
        py::module_::import("mtflib").attr("MultivariateTaylorFunction").attr("_IMPLEMENTATION") = backend_name;
    }, "Switch the backend implementation");
}
