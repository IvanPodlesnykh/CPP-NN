#include <iostream>
#include <fstream>
#include <random>
#include <Windows.h>
#include <time.h>
#include <math.h>
#include <tuple>

#define E_M 2.71828182845904523536

using namespace std;

random_device rd;
mt19937 gen(rd());
normal_distribution<float> ndist(0, 1);

double Sigmoid(double x) {
	return 1 / (1 + pow(E_M, -x));
}

double Sigmoid_prime(double x) {
	return Sigmoid(x) * (1 - Sigmoid(x));
}

void RandSort(int* m, int size) {
	int buf;
	int pos, pos2;
	for (int i = 0; i < rand() % 50 + 10; i++)
	{
		pos = rand() % size;
		pos2 = rand() % size;
		int buf = m[pos];
		m[pos] = m[pos2];
		m[pos2] = buf;
	}
}

class NeuralNetwork {
public:

	int num_layers; // Содержит значение числа слоев

	int* Sizes; // Массив, содержащий значение числа нейронов в каждом слое

	double** biases; // Массив смещений (bias) (1 измерение - номер слоя,
					 // 2 измерение - номер нейрона в данном слое)

	double*** weights; // Массив весов (1 измерение - номер слоя, в который
					   // входят веса, 2 измеренеие - номер нейрона из данного слоя
					   // 3 измерение - номер нейрона из предыдущего слоя)

	NeuralNetwork(int* sizes, const int number_of_layers) {

		num_layers = number_of_layers;
		biases = new double* [num_layers - 1];
		weights = new double** [num_layers - 1];
		Sizes = new int[number_of_layers];

		//Передаем значения sizes в поле Sizes
		for (int i = 0; i < number_of_layers; i++) {
			Sizes[i] = sizes[i];
		}

		//Задаем случайные отклонения (bias):
		for (int i = 0; i < num_layers - 1; i++) {
			biases[i] = new double[sizes[i + 1]];
			for (int j = 0; j < sizes[i + 1]; j++) {
				biases[i][j] = ndist(rd);
			}
		}
		//Задаем случайные веса
		for (int i = 0; i < num_layers - 1; i++) {
			weights[i] = new double* [sizes[i + 1]];
			for (int j = 0; j < sizes[i + 1]; j++) {
				weights[i][j] = new double[sizes[i]];
				for (int k = 0; k < sizes[i]; k++) {
					weights[i][j][k] = ndist(rd);
				}
			}
		}

	}

	~NeuralNetwork() {
		delete[] biases;
		delete[] weights;
		delete[] Sizes;
	}


	//Прямое распространение по нейронной сети, на выходе получаем вектор
	//выходных активаций нейронной сети, равный числу нейронов на выходе
	double* Feedforward(double* input) {
		double* result;
		double* temp;
		result = new double[Sizes[0]];
		for (int i = 0; i < Sizes[0]; i++) {
			result[i] = input[i];
		}
		for (int i = 0; i < num_layers - 1; i++) {
			temp = new double[Sizes[i + 1]];
			for (int j = 0; j < Sizes[i + 1]; j++) {
				temp[j] = biases[i][j];
				for (int k = 0; k < Sizes[i]; k++) {
					temp[j] += weights[i][j][k] * result[k];
				}
				temp[j] = Sigmoid(temp[j]);
			}
			delete[] result;
			result = new double[Sizes[i + 1]];
			for (int z = 0; z < Sizes[i + 1]; z++) {
				result[z] = temp[z];
			}
			delete[] temp;
		}
		return result;
	}

	//Алгоритм стохастического mini-batch градиентного спуска
	//training_data - двумерный массив, 1 размерность - наборы входных
	//данных + желаемый выход, 2 размерность - от нулевого до последнего
	//входы + желаемый выход (x0, x1, x2, x3 ... xN, y1, y2, y3...) 
	//для каждого набора
	//
	//len_training_data - число наборов входных данных
	//epochs - кол-во эпох (полных проходов через весь массив training_data туда и обратно)
	//mini_batch_size - размер одного мини-батча
	//eta - скорость обучения (learning rate)
	void SGD(double** training_data, int len_training_data, int epochs,
		int mini_batch_size, double eta) {

		double** buf;

		int number_of_batches = len_training_data / mini_batch_size;

		int* index_of_set = new int[len_training_data];

		//Создание массива индексов наборов
		for (int n = 0; n < len_training_data; n++) {
			index_of_set[n] = n;

		}

		//Цикл по эпохам
		for (int e = 0; e < epochs; e++) {

			//Перемешивание массива главных индексов
			RandSort(index_of_set, len_training_data);

			//Цикл по батчам
			for (int i = 0; i < number_of_batches; i++) {

				//Выделение памяти буферу
				buf = new double* [mini_batch_size];
				for (int alloc = 0; alloc < mini_batch_size; alloc++) {
					buf[alloc] = new double[Sizes[0] + Sizes[num_layers - 1]];
				}

				//Цикл передачи данных из мини батча в буфер
				for (int j = i; j < i + mini_batch_size; j++) {
					for (int k = 0; k < Sizes[0] + Sizes[num_layers - 1]; k++) {
						buf[j - i * mini_batch_size][k] = training_data[index_of_set[j]][k];
					}
				}

				//Обновление весов и смещений
				this->Update_mini_batch(buf, eta, mini_batch_size);

				//Освобождение памяти буфера
				delete[] buf;
			}
		}
	}

	//Метод, обновляющий веса и смещения после одного шага mini-batch SGD
	//mini_batch - массив, mini-batch набор входных данных
	//eta - скорость обучения (learning rate)
	void Update_mini_batch(double** mini_batch, double eta, int mini_batch_size) {

		double eps = eta / mini_batch_size;

		double** nabla_b;
		double*** nabla_w;

		//Выделяем память под nabla_b и nabla_w и заполням их нулями
		nabla_b = new double* [num_layers - 1];
		nabla_w = new double** [num_layers - 1];

		for (int i = 0; i < num_layers - 1; i++) {
			nabla_b[i] = new double[Sizes[i + 1]];
			for (int j = 0; j < Sizes[i + 1]; j++) {
				nabla_b[i][j] = 0;
			}
		}
		for (int i = 0; i < num_layers - 1; i++) {
			nabla_w[i] = new double* [Sizes[i + 1]];
			for (int j = 0; j < Sizes[i + 1]; j++) {
				nabla_w[i][j] = new double[Sizes[i]];
				for (int k = 0; k < Sizes[i]; k++) {
					nabla_w[i][j][k] = 0;
				}
			}
		}

		double** delta_nabla_b;
		double*** delta_nabla_w;

		double* xs;
		double* ys;

		//Цикл по наборам данных из одного mini_batch
		for (int batch_elem = 0; batch_elem < mini_batch_size; batch_elem++) {
			//Выделяем память под delta_nabla_b и delta_nabla_w
			delta_nabla_b = new double* [num_layers - 1];
			delta_nabla_w = new double** [num_layers - 1];

			for (int i = 0; i < num_layers - 1; i++) {
				delta_nabla_b[i] = new double[Sizes[i + 1]];
			}

			for (int i = 0; i < num_layers - 1; i++) {
				delta_nabla_w[i] = new double* [Sizes[i + 1]];
				for (int j = 0; j < Sizes[i]; j++) {
					delta_nabla_w[i][j] = new double[Sizes[i]];
				}
			}

			//Выделяем память и заполняем xs и ys
			xs = new double[Sizes[0]];
			ys = new double[Sizes[num_layers - 1]];
			for (int i = 0; i < Sizes[0]; i++) {
				xs[i] = mini_batch[batch_elem][i];
			}

			for (int j = Sizes[0]; j < Sizes[0] + Sizes[num_layers - 1]; j++) {
				ys[j] = mini_batch[batch_elem][j];
			}

			//Получаем результат backpropagation одного набора данных
			tuple<double**, double***> result = this->Backprop(xs, ys);

			delta_nabla_b = get<0>(result);

			delta_nabla_w = get<1>(result);

			//Инкрементируем nabla_b и nabla_w на результат backpropagation
			for (int i = 0; i < num_layers - 1; i++) {
				for (int j = 0; j < Sizes[i + 1]; j++) {
					nabla_b[i][j] += delta_nabla_b[i][j];
				}
			}
			for (int i = 0; i < num_layers - 1; i++) {
				for (int j = 0; j < Sizes[i + 1]; j++) {
					for (int k = 0; k < Sizes[i]; k++) {
						nabla_w[i][j][k] = delta_nabla_w[i][j][k];
					}
				}
			}

			delete[] xs, ys, delta_nabla_b, delta_nabla_w;
		}

		//Обновляем отклонения и веса
		for (int i = 0; i < num_layers - 1; i++) {
			for (int j = 0; j < Sizes[i + 1]; j++) {
				this->biases[i][j] = this->biases[i][j] - eps * nabla_b[i][j];
			}
		}
		for (int i = 0; i < num_layers - 1; i++) {
			for (int j = 0; j < Sizes[i + 1]; j++) {
				for (int k = 0; k < Sizes[i]; k++) {
					this->weights[i][j][k] = this->weights[i][j][k] - eps * nabla_w[i][j][k];
				}
			}
		}

	}

	//Алгоритм обратного распространения ошибки
	//Выводит tuple (nabla_b, nabla_w)
	tuple<double**, double***> Backprop(double* x, double* y) {

		//Выделение памяти под nabla_b и nabla_w
		double** nabla_b;
		double*** nabla_w;
		nabla_b = new double* [num_layers - 1];
		nabla_w = new double** [num_layers - 1];

		for (int i = 0; i < num_layers - 1; i++) {
			nabla_b[i] = new double[Sizes[i + 1]];
		}

		for (int i = 0; i < num_layers - 1; i++) {
			nabla_w[i] = new double* [Sizes[i + 1]];
			for (int j = 0; j < Sizes[i]; j++) {
				nabla_w[i][j] = new double[Sizes[i]];
			}
		}

		//Массив активаций
		double** activations;
		activations = new double* [num_layers];

		for (int i = 0; i < num_layers; i++) {
			activations[i] = new double[Sizes[i]];
		}

		//Запись входных значений в начало массива активаций
		for (int i = 0; i < Sizes[0]; i++) {
			activations[0][i] = x[i];
		}

		//Массив результатов сумматорной функции
		double** zs;
		zs = new double* [num_layers - 1];

		for (int i = 0; i < num_layers - 1; i++) {
			zs[i] = new double[Sizes[i + 1]];
		}

		//Заполнение массивов активаций и сумматорных функций
		for (int i = 0; i < num_layers - 1; i++) {
			for (int j = 0; j < Sizes[i + 1]; j++) {
				zs[i][j] = biases[i][j];
				for (int k = 0; k < Sizes[i]; k++) {
					zs[i][j] += weights[i][j][k] * activations[i][j];
				}
				activations[i + 1][j] = Sigmoid(zs[i][j]);
			}
		}

		//Массив дельт
		double** delta;
		delta = new double* [num_layers - 1];

		for (int i = 0; i < num_layers - 1; i++) {
			delta[i] = new double[Sizes[i + 1]];
		}

		//Заполняем дельты последнего слоя
		for (int i = 0; i < Sizes[num_layers - 1]; i++) {
			delta[num_layers - 2][i] = this->Cost_derivative(activations[num_layers - 1], y)[i] * Sigmoid_prime(zs[num_layers - 1][i]);
		}

		//Заполнение nabla_b последнего слоя
		for (int i = 0; i < Sizes[num_layers - 1]; i++) {
			nabla_b[num_layers - 2][i] = delta[num_layers - 2][i];
		}

		//Заполнение nabla_w последнего слоя
		for (int i = 0; i < Sizes[num_layers - 1]; i++) {
			for (int j = 0; j < Sizes[num_layers - 2]; j++) {
				nabla_w[num_layers - 2][i][j] = delta[num_layers - 2][i] * activations[num_layers - 2][j];
			}
		}

		//Заполняем весь массив дельт
		for (int i = num_layers - 2; i > 0; i--) {
			for (int j = 0; j < Sizes[i]; j++) {
				delta[i - 1][j] = 0;
				for (int k = 0; k < Sizes[i + 1]; j++) {
					delta[i - 1][j] += weights[i + 1][k][j] * delta[i][k];
				}
				delta[i - 1][j] = delta[i - 1][j] * Sigmoid_prime(zs[i - 1][j]);
			}
		}

		//Заполняем оставшийся массив nabla_b (все слои, кроме последнего)
		for (int i = 0; i < num_layers - 2; i++) {
			for (int j = 0; j < Sizes[i + 1]; j++) {
				nabla_b[i][j] = delta[i][j];
			}
		}

		//Заполняем оставшийся массив nabla_w (все слои, кроме последнего)
		for (int i = 0; i < num_layers - 2; i++) {
			for (int j = 0; j < Sizes[i + 1]; j++) {
				for (int k = 0; k < Sizes[i]; i++) {
					nabla_w[i][j][k] = delta[i][j] * activations[i][k];
				}
			}
		}

		//Создание кортежа в ответ
		tuple<double**, double***> result = make_tuple(nabla_b, nabla_w);
		return result;
	}

	//Производная целевой функции по активациям выходного слоя
	double* Cost_derivative(double* output_activations, double* y) {
		double* result = new double[Sizes[num_layers - 1]];
		for (int i = 0; i < Sizes[num_layers - 1]; i++) {
			result[i] = output_activations[i] - y[i];
		}
		return result;
	}

};

int main() {


	int size[4] = { 5, 5, 3, 1 };

	NeuralNetwork nn(size, 4);

	double samp[5] = { 1, 2, 3, 4, 5 };

	double* res;

	res = nn.Feedforward(samp);
	//_____________________________________________________

	cout << res[0];
	cout << res[1];

	return 0;
}
