#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mtf_data.hpp"
#include "biot_savart_ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mtf_cpp, m) {
    m.doc() = "C++ backend for MTFLibrary operations";

    py::class_<MtfData>(m, "MtfData")
        .def(py::init<>())
        .def(py::init<py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>())
        .def("to_dict", &MtfData::to_dict, "Converts the MtfData object to a dictionary of numpy arrays.")
        .def("add", &MtfData::add, "Adds another MtfData object.")
        .def("add_inplace", &MtfData::add_inplace, "Adds another MtfData object in-place.")
        .def("multiply", &MtfData::multiply, "Multiplies by another MtfData object.")
        .def("multiply_inplace", &MtfData::multiply_inplace, "Multiplies by another MtfData object in-place.")
        .def("compose", &MtfData::compose, "Composes with a dictionary of MtfData objects.")
        .def("negate", &MtfData::negate, "Negates the MtfData object.")
        .def("power", &MtfData::power, "Raises the MtfData object to a power.");

    m.def("add_mtf_cpp", &add_mtf_cpp, "A function that adds two MTFs using C++");
    m.def("multiply_mtf_cpp", &multiply_mtf_cpp, "A function that multiplies two MTFs using C++");
    m.def("extract_coefficient_cpp", &extract_coefficient_cpp, "A function that extracts a coefficient from an MTF using C++");
    m.def("biot_savart_core_cpp", &biot_savart_core_cpp, "A function that computes the Biot-Savart law using C++");
    m.def("biot_savart_from_numpy", &biot_savart_from_numpy, "A function that computes the Biot-Savart law from numpy arrays");
    m.def("biot_savart_from_flat_numpy", &biot_savart_from_flat_numpy, "A function that computes the Biot-Savart law from flat numpy arrays");

    m.def("sqrt_taylor_1D_expansion", &sqrt_taylor_1D_expansion, "1D Taylor expansion of sqrt(1+u)");
    m.def("isqrt_taylor_1D_expansion", &isqrt_taylor_1D_expansion, "1D Taylor expansion of isqrt(1+u)");
    m.def("inverse_taylor_1D_expansion", &inverse_taylor_1D_expansion, "1D Taylor expansion of 1/(1-u)");
}
