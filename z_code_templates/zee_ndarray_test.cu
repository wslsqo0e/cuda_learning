#include "zee_ndarray.h"

using namespace zee;

void test_transpose() {
  NDArray<float> m(16, 8);
  m.random_init();
  NDArray<float> n = m.transpose();

  for (size_t i = 0; i < m.M(); i++) {
    for (size_t j = 0; j < m.N(); j++) {
      if (m(i, j) != n(j, i)) {
        throw std::runtime_error("transpose failed");
      }
    }
  }

  printf("test transpose success!\n");
}

int main() {
    test_transpose();
}
