#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void mat_mul(const float *X, const float *Y, float *Z,
        int a, int b, int c) 
{
  // X:(a, b), Y:(b, c), Z:(a, c)
  for (int i = 0; i < a; i ++) {
    for (int j = 0; j < c; j ++) {
      Z[i * c + j] = 0;
      for (int k = 0 ; k < b; k ++) {
        Z[i * c + j] += X[i * b + k] * Y[k * c + j];
      }
    }
  }
}

void transpose(const float *X, float *Y,
         int m, int n)
{
  for (int i = 0; i < m; i ++) {
    for (int j = 0; j < n; j ++) {
      // Y[j][i] = X[i][j];
      Y[j * m + i] = X[i * n + j];
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch_size)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_examples = m;
    int batch = (num_examples + batch_size - 1) / batch_size;
    for (int i = 0; i < batch; i ++) {
      int start = i * batch_size;

      const float *x = &X[start * n];
      float *Z = new float[batch_size * k];
      mat_mul(x, theta, Z, batch_size, n, k);
      // Z = np.exp(x @ theta)
      for (int i = 0; i < batch_size * k; i ++) {
        Z[i] = expf(Z[i]);
      }
      // Z = Z / np.sum(Z, axis=1, keepdims=True)
      for (int i = 0; i < batch_size; i ++) {
        float sum = 0;
        for (int j = 0; j < k; j ++) {
          sum += Z[i * k + j];
        }
        for (int j = 0; j < k; j ++) {
          Z[i * k + j] /= sum;
        }
      }
      // grad = x.T @ (Z - Y) / batch_size
      // Z -= Y
      for (int i = 0; i < batch_size; i ++) {
        Z[i * k + y[start + i]] --;
      }
      // grad = x.T @ Z / batch_size
      float *x_T = new float[n * batch_size];
      transpose(x, x_T, batch_size, n);
      float *grad = new float[n * k];
      mat_mul(x_T, Z, grad, n, batch_size, k);
      // theta -= lr * grad
      for (int i = 0; i < n * k; i ++) {
        theta[i] -= lr * grad[i] / (float)batch_size;
      }

      // delete Z, x_T, grad
      delete[] Z;
      delete[] x_T;
      delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
