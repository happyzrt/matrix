#include <iostream>
#include "ATen/ATen.h"

int main(int argc, char* argv[])
{
    at::Tensor a = at::ones({2, 2}, at::kInt);
    at::Tensor b = at::randn({2, 2});
    auto c = a + b.to(at::kInt);
    return 0;
}
