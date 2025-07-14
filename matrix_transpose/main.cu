#include "zee_ndarray.h"



using namespace zee;

int main() {
  NDArray<float> a(3, 4);
  a.random_init();
  NDArray<float> b = a.transpose();
  a.print();
  printf("============\n");
  b.print();

  printf("Hello World\n");
}
