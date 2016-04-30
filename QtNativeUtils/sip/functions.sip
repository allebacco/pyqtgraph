
%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

%ModuleCode
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL numpy_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "functions.h"
#include "internal/Numpy.h"
%End


QPainterPath arrayToQPath(PyObject* x, PyObject* y, PyObject* connect=Py_None);
%MethodCode
    PyObject* pyX = a0;
    PyObject* pyY = a1;
    PyObject* pyConnect = a2;

    try{

        NDArray x(pyX);
        NDArray y(pyY);
        QString connect;
        if(PyString_Check(pyConnect))
            connect = PyString_AsString(pyConnect);

        if(x.dtype()!=NPY_FLOAT64 || y.dtype()!=NPY_FLOAT64)
            throw std::runtime_error("x and y has not dtype float64");

        if(x.ndims()!=1 || y.ndims()!=1)
            throw std::runtime_error("x and y has ndims != 1");

        if(x.shape(0)!=y.shape(0))
            throw std::runtime_error("x and y different shape");

        sipRes = new QPainterPath(arrayToQPath(x.data<double>(), y.data<double>(), x.shape(0), connect));

    } catch(const std::runtime_error& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        return nullptr;
    }


%End
