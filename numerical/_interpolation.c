/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>

static const double EPS = 1.e-12;

static inline _Bool isEq(const double a, const double b)
{
    return -(a - b) < EPS && (a - b) < EPS;
}

static inline _Bool isGeq(const double a, const double b)
{
    return a > b || isEq(a, b);
}

static inline _Bool isLeq(const double a, const double b)
{
    return a < b || isEq(a, b);
}

static inline double _bSpline(double t)
{
    if (t > -1 && isLeq(t, 0))
        return t + 1;
    else if (t > 0 && t < 1)
        return 1 - t;
    else
        return 0;
}


static PyObject* line_interpolate_1d(PyObject* self, PyObject* args, PyObject* keywds)
{
    int n;
    npy_intp* shape;
    double a, b;

    PyArrayObject *in_array_x, *in_array_uniform_tabs;
    PyObject      *out_array;
    NpyIter *in_iter_x, *in_iter_uniform_tabs;
    NpyIter *out_iter;
    NpyIter_IterNextFunc *in_iternext_x, *in_iternext_uniform_tabs;
    NpyIter_IterNextFunc *out_iternext;

    /*  parse single numpy array argument */
    static char *kwlist[] = {"x", "uniform_tabs", "limits", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!(dd)", kwlist, \
                          &PyArray_Type, &in_array_x, &PyArray_Type, &in_array_uniform_tabs, \
                          &a, &b))
        return NULL;

    shape = PyArray_SHAPE(in_array_uniform_tabs);
    if (shape == NULL)
        return NULL;

    n = (int)shape[0];

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array_x, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    /*  create the iterators */
    in_iter_x = NpyIter_New(in_array_x, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_uniform_tabs = NpyIter_New(in_array_uniform_tabs, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    if (in_iter_x == NULL || in_iter_uniform_tabs == NULL)
        goto fail;

    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
                          NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (out_iter == NULL) {
        NpyIter_Deallocate(in_iter_x);
        NpyIter_Deallocate(in_iter_uniform_tabs);
        goto fail;
    }

    in_iternext_x = NpyIter_GetIterNext(in_iter_x, NULL);
    in_iternext_uniform_tabs = NpyIter_GetIterNext(in_iter_uniform_tabs, NULL);

    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    if (in_iternext_x == NULL || in_iternext_uniform_tabs == NULL || \
        out_iternext == NULL) {
        NpyIter_Deallocate(in_iter_x);
        NpyIter_Deallocate(in_iter_uniform_tabs);
        NpyIter_Deallocate(out_iter);
        goto fail;
    }
    double ** in_dataptr_x = (double **) NpyIter_GetDataPtrArray(in_iter_x);
    double ** in_dataptr_uniform_tabs = (double **) NpyIter_GetDataPtrArray(in_iter_uniform_tabs);
    double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);

    /*  iterate over the arrays */
    register double sum = 0.0;

    int i;
    do
    {
        i = 0;
        sum = 0.0;
        do
        {
            sum += **in_dataptr_uniform_tabs * _bSpline(((n - 1) * (**in_dataptr_x - a)) / (b - a) - i);
            ++i;
        } while (in_iternext_uniform_tabs(in_iter_uniform_tabs));
        **out_dataptr = sum;

        NpyIter_Reset(in_iter_uniform_tabs, NULL);
    } while (in_iternext_x(in_iter_x) && out_iternext(out_iter));


    /*  clean up and return the result */
    NpyIter_Deallocate(in_iter_x);
    NpyIter_Deallocate(in_iter_uniform_tabs);
    NpyIter_Deallocate(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}

static PyObject* line_interpolate_2d(PyObject* self, PyObject* args, PyObject* keywds)
{
    int n1, n2;
    npy_intp* shape;
    double a1, b1, a2, b2;

    PyArrayObject *in_array_x1, *in_array_x2, *in_array_uniform_tabs;
    PyObject      *out_array;
    NpyIter *in_iter_x1, *in_iter_x2; //, *in_iter_uniform_tabs;
    NpyIter *out_iter;
    NpyIter_IterNextFunc *in_iternext_x1, *in_iternext_x2; //, *in_iternext_uniform_tabs;
    NpyIter_IterNextFunc *out_iternext;

    /*  parse single numpy array argument */
    static char *kwlist[] = {"x1", "x2", "uniform_tabs", "limits", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!(dddd)", kwlist, \
                          &PyArray_Type, &in_array_x1, &PyArray_Type, &in_array_x2, \
                          &PyArray_Type, &in_array_uniform_tabs, \
                          &a1, &b1, &a2, &b2))
        return NULL;

    shape = PyArray_SHAPE(in_array_uniform_tabs);
    if (shape == NULL)
        return NULL;

    n1 = (int)shape[0];
    n2 = (int)shape[1];

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array_x1, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    /*  create the iterators */
    in_iter_x1 = NpyIter_New(in_array_x1, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_x2 = NpyIter_New(in_array_x2, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
//    in_iter_uniform_tabs = NpyIter_New(in_array_uniform_tabs, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    if (in_iter_x1 == NULL || in_iter_x2 == NULL)
        goto fail;

    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
                          NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (out_iter == NULL) {
        NpyIter_Deallocate(in_iter_x1);
        NpyIter_Deallocate(in_iter_x2);
//        NpyIter_Deallocate(in_iter_uniform_tabs);
        NpyIter_Deallocate(out_iter);
        goto fail;
    }

    in_iternext_x1 = NpyIter_GetIterNext(in_iter_x1, NULL);
    in_iternext_x2 = NpyIter_GetIterNext(in_iter_x2, NULL);
    // in_iternext_uniform_tabs = NpyIter_GetIterNext(in_iter_uniform_tabs, NULL);

    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    if (in_iternext_x1 == NULL || in_iternext_x2 == NULL || \
        out_iternext == NULL) {
        NpyIter_Deallocate(in_iter_x1);
        NpyIter_Deallocate(in_iter_x2);
//        NpyIter_Deallocate(in_iter_uniform_tabs);
        NpyIter_Deallocate(out_iter);
        goto fail;
    }
    double ** in_dataptr_x1 = (double **) NpyIter_GetDataPtrArray(in_iter_x1);
    double ** in_dataptr_x2 = (double **) NpyIter_GetDataPtrArray(in_iter_x2);
//    double ** in_dataptr_uniform_tabs = (double **) NpyIter_GetDataPtrArray(in_iter_uniform_tabs);
    double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);

    /*  iterate over the arrays */
    register double sum = 0.0;
    double *aij;
    do
    {
        sum = 0.0;
        for (int i = 0; i < n1; ++i)
        {
            for (int j = 0; j < n2; ++j)
            {
                aij = (double*)PyArray_GETPTR2(in_array_uniform_tabs, i, j);
                sum += *aij * \
                    _bSpline(((n1 - 1) * (**in_dataptr_x1 - a1)) / (b1 - a1) - i) * \
                    _bSpline(((n2 - 1) * (**in_dataptr_x2 - a2)) / (b2 - a2) - j);
            }
        }

        **out_dataptr = sum;

        // NpyIter_Reset(in_iter_uniform_tabs, NULL);
    } while (in_iternext_x1(in_iter_x1) && in_iternext_x2(in_iter_x2) && out_iternext(out_iter));


    /*  clean up and return the result */
    NpyIter_Deallocate(in_iter_x1);
    NpyIter_Deallocate(in_iter_x2);
//    NpyIter_Deallocate(in_iter_uniform_tabs);
    NpyIter_Deallocate(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}


static PyObject* line_interpolate_3d(PyObject* self, PyObject* args, PyObject* keywds)
{
    int n1, n2, n3;
    npy_intp* shape;
    double a1, b1, a2, b2, a3, b3;

    PyArrayObject *in_array_x1, *in_array_x2, *in_array_x3, *in_array_uniform_tabs;
    PyObject      *out_array;
    NpyIter *in_iter_x1, *in_iter_x2, *in_iter_x3, *in_iter_uniform_tabs;
    NpyIter *out_iter;
    NpyIter_IterNextFunc *in_iternext_x1, *in_iternext_x2, *in_iternext_x3, *in_iternext_uniform_tabs;
    NpyIter_IterNextFunc *out_iternext;

    /*  parse single numpy array argument */
    static char *kwlist[] = {"x1", "x2", "x3", "uniform_tabs", "limits", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!(dddddd)", kwlist, \
                          &PyArray_Type, &in_array_x1, &PyArray_Type, &in_array_x2, &PyArray_Type, &in_array_x3, \
                          &PyArray_Type, &in_array_uniform_tabs, \
                          &a1, &b1, &a2, &b2, &a3, &b3))
        return NULL;

    shape = PyArray_SHAPE(in_array_uniform_tabs);
    if (shape == NULL)
        return NULL;

    n1 = (int)shape[0];
    n2 = (int)shape[1];
    n3 = (int)shape[2];

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array_x1, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    /*  create the iterators */
    in_iter_x1 = NpyIter_New(in_array_x1, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_x2 = NpyIter_New(in_array_x2, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_x3 = NpyIter_New(in_array_x3, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_uniform_tabs = NpyIter_New(in_array_uniform_tabs, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    if (in_iter_x1 == NULL || in_iter_x2 == NULL || in_iter_x3 == NULL || in_iter_uniform_tabs == NULL)
        goto fail;

    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
                          NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (out_iter == NULL) {
        NpyIter_Deallocate(in_iter_x1);
        NpyIter_Deallocate(in_iter_x2);
        NpyIter_Deallocate(in_iter_x3);
        NpyIter_Deallocate(in_iter_uniform_tabs);
        goto fail;
    }

    in_iternext_x1 = NpyIter_GetIterNext(in_iter_x1, NULL);
    in_iternext_x2 = NpyIter_GetIterNext(in_iter_x2, NULL);
    in_iternext_x3 = NpyIter_GetIterNext(in_iter_x3, NULL);
    in_iternext_uniform_tabs = NpyIter_GetIterNext(in_iter_uniform_tabs, NULL);

    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    if (in_iternext_x1 == NULL || in_iternext_x2 == NULL || in_iternext_x3 == NULL || in_iternext_uniform_tabs == NULL || \
        out_iternext == NULL) {
        NpyIter_Deallocate(in_iter_x1);
        NpyIter_Deallocate(in_iter_x2);
        NpyIter_Deallocate(in_iter_x3);
        NpyIter_Deallocate(in_iter_uniform_tabs);
        NpyIter_Deallocate(out_iter);
        goto fail;
    }
    double ** in_dataptr_x1 = (double **) NpyIter_GetDataPtrArray(in_iter_x1);
    double ** in_dataptr_x2 = (double **) NpyIter_GetDataPtrArray(in_iter_x2);
    double ** in_dataptr_x3 = (double **) NpyIter_GetDataPtrArray(in_iter_x3);
    double ** in_dataptr_uniform_tabs = (double **) NpyIter_GetDataPtrArray(in_iter_uniform_tabs);
    double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);

    /*  iterate over the arrays */
    register double sum = 0.0;
    do
    {
        sum = 0.0;
        for (int i = 0; i < n1; ++i)
        {
            for (int j = 0; j < n2; ++j)
            {
                for (int k = 0; k < n3; ++k)
                {
                    sum += **in_dataptr_uniform_tabs * \
                        _bSpline(((n1 - 1) * (**in_dataptr_x1 - a1)) / (b1 - a1) - i) * \
                        _bSpline(((n2 - 1) * (**in_dataptr_x2 - a2)) / (b2 - a2) - j) * \
                        _bSpline(((n3 - 1) * (**in_dataptr_x3 - a3)) / (b3 - a3) - k);
                    in_iternext_uniform_tabs(in_iter_uniform_tabs);
                }
            }
        }

        **out_dataptr = sum;

        NpyIter_Reset(in_iter_uniform_tabs, NULL);
    } while (in_iternext_x1(in_iter_x1) && in_iternext_x2(in_iter_x2) && \
             in_iternext_x3(in_iter_x3) && out_iternext(out_iter));


    /*  clean up and return the result */
    NpyIter_Deallocate(in_iter_x1);
    NpyIter_Deallocate(in_iter_x2);
    NpyIter_Deallocate(in_iter_x3);
    NpyIter_Deallocate(in_iter_uniform_tabs);
    NpyIter_Deallocate(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}


/*  define functions in module */
static PyMethodDef InterpolationFunctions[] =
{
     {"line_interpolate_1d", (PyCFunction)line_interpolate_1d, METH_VARARGS|METH_KEYWORDS, ""},
     {"line_interpolate_2d", (PyCFunction)line_interpolate_2d, METH_VARARGS|METH_KEYWORDS, ""},
     {"line_interpolate_3d", (PyCFunction)line_interpolate_3d, METH_VARARGS|METH_KEYWORDS, ""},
     {NULL, NULL, 0, NULL}
};

static struct PyModuleDef interpolationmodule = {
    PyModuleDef_HEAD_INIT,
    "_interpolation",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    InterpolationFunctions
};

PyMODINIT_FUNC
PyInit__interpolation(void)
{
    import_array();
    PyObject *m;
    m = PyModule_Create(&interpolationmodule);
    if (!m)
        return NULL;

    return m;
}
