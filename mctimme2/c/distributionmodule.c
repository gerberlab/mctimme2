#include <Python.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>


static PyObject *
distribution_traj_logpdf(PyObject *self, PyObject *args) {
  double rhs0;
  double rhs1;
  double p;

  double traj0;
  double growth_rate;
  double persist0;
  double dt0;
  double self_interact;
  double pert_traj0;
  double traj1;
  double persist1;
  double dt1;
  double pert_traj1;

  double traj_var0;
  double traj_var1;
  double traj2;

  double aux_var;
  double q1;

  if (!PyArg_ParseTuple(args, "ddddddddddddddd", &traj0, &traj1, &traj2, &growth_rate, &self_interact,
                       &dt0, &dt1, &pert_traj0, &pert_traj1, &persist0, &persist1, &traj_var0, &traj_var1, &aux_var, &q1))
    return NULL;

  rhs0 = traj0 + growth_rate * (1 + persist0) * traj0 * dt0 + self_interact * pow(traj0,2) * dt0 + growth_rate * pert_traj0 * traj0 * dt0;
  rhs1 = traj1 + growth_rate * (1 + persist1) * traj1 * dt1 + self_interact * pow(traj1,2) * dt1 + growth_rate * pert_traj1 * traj1 * dt1;

  p = -0.9189385332046727 - log(sqrt(traj_var0 * dt0)) - pow(traj1-rhs0,2) / (2*traj_var0 * dt0)
      -0.9189385332046727 - log(sqrt(traj_var1 * dt1)) - pow(traj2-rhs1,2) / (2*traj_var1 * dt1)
      -0.9189385332046727 - log(sqrt(aux_var)) - pow(traj1-q1,2) / (2*aux_var);

  return PyFloat_FromDouble(p);

}


static PyObject *
distribution_normal_logpdf(PyObject *self, PyObject *args) {
  double x;
  double loc;
  double scale;
  if (!PyArg_ParseTuple(args, "ddd", &x, &loc, &scale))
        return NULL;
 return PyFloat_FromDouble(-0.9189385332046727 - log(scale) - pow(x-loc,2) / (2*pow(scale,2)));
}


static PyObject *
distribution_noise_model(PyObject *self, PyObject *args) {
  double phi;
  double w;
  double r;

  double k;
  double m;

  double max;
  double sum;

  if (!PyArg_ParseTuple(args, "dd", &k, &m))
    return NULL;

  if (m == 0 && k == 0)
    return 0;
  if (m == 0)
    return PyFloat_FromDouble(-INFINITY);

  double logp1;
  w = 0.095;
  phi = 0.00144;
  r = 1/phi;
  logp1 = log(w) + lgamma(k + r) - lgamma(k + 1) - lgamma(r) + r * (log(r) - log(r+m)) + k * (log(m) - log(r+m));

  double logp2;
  w = 0.247;
  phi = 0.00268;
  r = 1/phi;
  logp2 = log(w) + lgamma(k + r) - lgamma(k + 1) - lgamma(r) + r * (log(r) - log(r+m)) + k * (log(m) - log(r+m));

  double logp3;
  w = 0.63;
  phi = 0.000742;
  r = 1/phi;
  logp3 = log(w) + lgamma(k + r) - lgamma(k + 1) - lgamma(r) + r * (log(r) - log(r+m)) + k * (log(m) - log(r+m));

  double logp4;
  w = 0.028;
  phi = 0.0112;
  r = 1/phi;
  logp4 = log(w) + lgamma(k + r) - lgamma(k + 1) - lgamma(r) + r * (log(r) - log(r+m)) + k * (log(m) - log(r+m));

  max = -INFINITY;
  if (logp1 > max)
    max = logp1;
  if (logp2 > max)
    max = logp2;
  if (logp3 > max)
    max = logp3;
  if (logp4 > max)
    max = logp4;

  sum = max + log(exp(logp1-max) + exp(logp2-max) + exp(logp3-max) + exp(logp4-max));

  return PyFloat_FromDouble(sum);

}


static PyObject *
distribution_truncated_normal_logpdf(PyObject *self, PyObject *args) {
  double x;
  double loc;
  double scale;
  double a;
  double b;

  double a1;
  double a2;
  double a3;

  if (!PyArg_ParseTuple(args, "ddddd", &x, &loc, &scale, &a, &b))
        return NULL;
  if (x < a || a > b){
    return PyFloat_FromDouble(-INFINITY);
  }
  if (x > b){
    return PyFloat_FromDouble(-INFINITY);
  }
  a1 = (x - loc) / scale;
  a2 = (b - loc) / scale;
  a3 = (a - loc) / scale;

  return PyFloat_FromDouble(M_LN2 - 0.9189385332046727 - 0.5*pow(a1,2) - log(scale) - log(erf(a2/M_SQRT2) - erf(a3/M_SQRT2)));
}

static PyObject *
distribution_lognormal_logpdf(PyObject *self, PyObject *args) {
  double x;
  double mu;
  double sigma;

  if (!PyArg_ParseTuple(args, "ddd", &x, &mu, &sigma))
    return NULL;
  if (x <= 0) {
    return PyFloat_FromDouble(-INFINITY);
  }
  return PyFloat_FromDouble(-0.9189385332046727 - log(x) - log(sigma) - pow(log(x) - mu, 2) / (2*pow(sigma,2)));

}

static PyMethodDef DistributionMethods[] = {
  {"normal_logpdf", distribution_normal_logpdf, METH_VARARGS, "logpdf"},
  {"truncated_normal_logpdf", distribution_truncated_normal_logpdf, METH_VARARGS, "logpdf"},
  {"lognormal_logpdf", distribution_lognormal_logpdf, METH_VARARGS, "lognormal logpdf"},
  {"traj_logpdf", distribution_traj_logpdf, METH_VARARGS, "traj logpdf"},
  {"noise_model", distribution_noise_model, METH_VARARGS, "noise model"},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef distributionmodule = {
  PyModuleDef_HEAD_INIT,
  "distribution",
  NULL,
  -1,
  DistributionMethods
};

PyMODINIT_FUNC
PyInit_distribution(void){
  return PyModule_Create(&distributionmodule);
}

int
main(int argc, char *argv[]) {
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL) {
      fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
      exit(1);
  }

  /* Add a built-in module, before Py_Initialize */
  PyImport_AppendInittab("distribution", PyInit_distribution);

  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Optionally import the module; alternatively,
     import can be deferred until the embedded script
     imports it. */
  PyImport_ImportModule("distribution");

  PyMem_RawFree(program);
  return 0;
}
