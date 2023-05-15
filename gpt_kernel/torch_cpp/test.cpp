#include <iostream>
#include <torch/torch.h>

using std::cout;
using std::endl;

int main() {

    torch::Tensor x1;

    if(!x1.defined()) {

        cout << "x1 is not defined\n";
    }

    x1 = torch::randn({3, 3});

    if(!x1.defined()) {

        cout << "x1 is not defined\n";
    }

    else {

        cout << "x1: " << endl;
    }

    return 0;

}