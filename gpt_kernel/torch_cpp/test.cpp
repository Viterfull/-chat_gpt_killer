#include <iostream>
#include <string>
#include <typeinfo>
#include <torch/torch.h>

using std::cout;
using std::endl;

int main() {

    const c10::DeviceType device1 = torch::kCUDA;
    cout << device1;

    return 0;

}