#define _GNU_SOURCE
#define Py_USING_UNICODE
#include <Python.h>
#define  NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "pycomm.h"

static int pyt_check_np_array_1d(PyObject *p, int size, const char *msg)
{
    int nd, np_type; npy_intp *dims;
    nd = PyArray_NDIM((PyArrayObject*)p);
    dims = PyArray_DIMS((PyArrayObject*)p);
    np_type = PyArray_TYPE((PyArrayObject*)p);
    if(nd!=1 || dims[0]!=size) {
        printf("%s: %s wrong dimension or size!", __func__, msg);
        return -1;
    }
    if(np_type!=NPY_FLOAT) {
        printf("%s: %s wrong type, requires float!", __func__, msg);
        return -1;
    }
    return 0;
}

/**
 * @brief b = toeplitz_mat04(a, f), 4x4 toeplitz
 * */
static PyObject * pyt_toeplitz_mat04(PyObject __attribute__((unused)) *self, PyObject *args)
{
    PyObject *obj_a, *obj_f; float *buf_a, *buf_f;
    if(!PyArg_ParseTuple(args, "OO", &obj_a, &obj_f))
        return NULL;

    //check object a, must be length of 7
    if(pyt_check_np_array_1d(obj_a, 7, "input array")) 
        return Py_False;

    //check object f, must be length of 4
    if(pyt_check_np_array_1d(obj_f, 4, "input filter"))
        return Py_False;

    //create new object b for output!
    PyObject *obj_b; float *buf_b;
    int nd=1; npy_intp dims[] = {4};
    obj_b = PyArray_SimpleNew(nd, dims, NPY_FLOAT);
    buf_a = PyArray_DATA((PyArrayObject*)obj_a);
    buf_f = PyArray_DATA((PyArrayObject*)obj_f);
    buf_b = PyArray_DATA((PyArrayObject*)obj_b);
    Toeplitz_mat4(buf_a, buf_f, buf_b);
    return obj_b;
}

static PyMethodDef pyt_Methods[] = {
    {"mat04",     pyt_toeplitz_mat04,      METH_VARARGS, "Toeplitz 4x4 Matrix"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

//static PyObject *PydbError;

PyMODINIT_FUNC initpytoep(void)
{
    PyObject *m;
    m = Py_InitModule("pytoep", pyt_Methods);
    if (m == NULL)
        return;
    import_array();
}
