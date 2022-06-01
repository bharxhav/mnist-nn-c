#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

int main()
{
	srand(time(NULL));

	// TRAINING
	int number_imgs = 10000;
	Img **imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
	NeuralNetwork *net = network_create(784, 300, 10);
	network_train_batch_imgs(net, imgs, number_imgs);
	network_save(net, "testing_net");
	return 0;
}