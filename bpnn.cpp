#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>

#define INNODE 2
#define HIDDENNODE 4
#define OUTNODE 1

double rate = 0.8;
double threshold = 1e-4;
size_t mosttimes = 1e6;

typedef struct Sample{
    std::vector<double> in, out;
} Sample;

typedef struct Node {
    double value{}, bias{}, bias_delta{};
    std::vector<double> weight, weight_delta;
} Node;


namespace utils {

    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::vector<double> getFileData(std::string &filename) {
        std::vector<double> res;

        std::ifstream in(filename);
        if (in.is_open()) {
            while (!in.eof()) {
                double buffer;
                in >> buffer;
                res.push_back(buffer);
            }
            in.close();
        } else {
            std::cout << "Error in reading " << filename << std::endl;
        }

        return res;
    }

    std::vector<Sample> getTrainData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE + OUTNODE) {
            Sample sample;
            for (size_t t = 0; t < INNODE; t++) {
                sample.in.push_back(buffer[i + t]);
            }
            for (size_t t = 0; t < OUTNODE; t++) {
                sample.out.push_back(buffer[i + INNODE + t]);
            }
            res.push_back(sample);
        }

        return res;
    }

    std::vector<Sample> getTestData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE) {
            Sample sample;
            for (size_t t = 0; t < INNODE; t++) {
                sample.in.push_back(buffer[i + t]);
            }
            res.push_back(sample);
        }

        return res;
    }
}

Node *inputLayer[INNODE], *hiddenLayer[HIDDENNODE], *outputLayer[OUTNODE];

inline void init() {
    std::mt19937 rd;
    rd.seed(std::random_device()());
    std::uniform_real_distribution<double> distribution(-1, 1);

    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i] = new Node();
        for (size_t j = 0; j < HIDDENNODE; j++) {
            ::inputLayer[i]->weight.push_back(distribution(rd));
            ::inputLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < HIDDENNODE; i++) {
        ::hiddenLayer[i] = new Node();
        ::hiddenLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < OUTNODE; j++) {
            ::hiddenLayer[i]->weight.push_back(distribution(rd));
            ::hiddenLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < OUTNODE; i++) {
        ::outputLayer[i] = new Node();
        ::outputLayer[i]->bias = distribution(rd);
    }
}

inline void reset_delta() {
    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i]->weight_delta.assign(::inputLayer[i]->weight_delta.size(), 0.f);
    }
    for (size_t i = 0; i < HIDDENNODE; i++) {
        ::hiddenLayer[i]->bias_delta = 0.f;
        ::hiddenLayer[i]->weight_delta.assign(::hiddenLayer[i]->weight_delta.size(), 0.f);
    }
    for (size_t i = 0; i < OUTNODE; i++) {
        ::outputLayer[i]->bias_delta = 0.f;
    }
}

int main(int argc, char *argv[]) {

    init();

    std::vector<Sample> train_data = utils::getTrainData("traindata.txt");

    for (size_t times = 0; times < mosttimes; times++) {

        reset_delta();

        double error_max = 0;

        for (auto & idx : train_data) {

            for (size_t i = 0; i < INNODE; i++) {
                ::inputLayer[i]->value = idx.in[i];
            }

            // 正向传播
            for (size_t j = 0; j < HIDDENNODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < INNODE; i++) {
                    sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
                }
                sum -= ::hiddenLayer[j]->bias;
                ::hiddenLayer[j]->value = utils::sigmoid(sum);
            }

            for (size_t j = 0; j < OUTNODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < HIDDENNODE; i++) {
                    sum += ::hiddenLayer[i]->value * ::hiddenLayer[i]->weight[j];
                }
                sum -= ::outputLayer[j]->bias;
                ::outputLayer[j]->value = utils::sigmoid(sum);
            }

            // 计算损失函数
            double error = 0;
            for (size_t i = 0; i < OUTNODE; i++) {
                double tmp = std::fabs(::outputLayer[i]->value - idx.out[i]);
                error += tmp * tmp / 2;
            }

            error_max = std::max(error_max, error);

            // 反向传播
            for (size_t i = 0; i < OUTNODE; i++) {
                double bias_delta = -(idx.out[i] - ::outputLayer[i]->value) *
                                    ::outputLayer[i]->value * (1.0 - ::outputLayer[i]->value);
                ::outputLayer[i]->bias_delta += bias_delta;
            }

            for (size_t i = 0; i < HIDDENNODE; i++) {
                for (size_t j = 0; j < OUTNODE; j++) {
                    double weight_delta = (idx.out[j] - ::outputLayer[j]->value) *
                                          ::outputLayer[j]->value * (1.0 - ::outputLayer[j]->value) *
                                          ::hiddenLayer[i]->value;
                    ::hiddenLayer[i]->weight_delta[j] += weight_delta;
                }
            }

            for (size_t i = 0; i < HIDDENNODE; i++) {
                double sum = 0;
                for (size_t j = 0; j < OUTNODE; j++) {
                    sum += -(idx.out[j] - ::outputLayer[j]->value) *
                           ::outputLayer[j]->value * (1.0 - ::outputLayer[j]->value) *
                           ::hiddenLayer[i]->weight[j];
                }
                ::hiddenLayer[i]->bias_delta +=
                        sum * ::hiddenLayer[i]->value * (1.0 - ::hiddenLayer[i]->value);
            }

            for (size_t i = 0; i < INNODE; i++) {
                for (size_t j = 0; j < HIDDENNODE; j++) {
                    double sum = 0;
                    for (size_t k = 0; k < OUTNODE; k++) {
                        sum += (idx.out[k] - ::outputLayer[k]->value) *
                               ::outputLayer[k]->value * (1.0 - ::outputLayer[k]->value) *
                               ::hiddenLayer[j]->weight[k];
                    }
                    ::inputLayer[i]->weight_delta[j] +=
                            sum * ::hiddenLayer[j]->value * (1.0 - ::hiddenLayer[j]->value) * ::inputLayer[i]->value;
                }
            }

        }

        if (error_max < threshold) {
            std::cout << "Success with " << times + 1 << " times training." << std::endl;
            std::cout << "Maximum error: " << error_max << std::endl;
            break;
        }

        auto train_data_size = double(train_data.size());

        for (size_t i = 0; i < INNODE; i++) {
            for (size_t j = 0; j < HIDDENNODE; j++) {
                ::inputLayer[i]->weight[j] += rate * ::inputLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < HIDDENNODE; i++) {
            ::hiddenLayer[i]->bias += rate * ::hiddenLayer[i]->bias_delta / train_data_size;
            for (size_t j = 0; j < OUTNODE; j++) {
                ::hiddenLayer[i]->weight[j] += rate * ::hiddenLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < OUTNODE; i++) {
            ::outputLayer[i]->bias += rate * ::outputLayer[i]->bias_delta / train_data_size;
        }

    }

    std::vector<Sample> test_data = utils::getTestData("testdata.txt");

    for (size_t idx = 0; idx < test_data.size(); idx++) {

        for (size_t i = 0; i < INNODE; i++) {
            inputLayer[i]->value = test_data[idx].in[i];
        }

        for (size_t j = 0; j < HIDDENNODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < INNODE; i++) {
                sum += inputLayer[i]->value * inputLayer[i]->weight[j];
            }
            sum -= ::hiddenLayer[j]->bias;
            ::hiddenLayer[j]->value = utils::sigmoid(sum);
        }

        for (size_t j = 0; j < OUTNODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < HIDDENNODE; i++) {
                sum += ::hiddenLayer[i]->value * ::hiddenLayer[i]->weight[j];
            }
            sum -= ::outputLayer[j]->bias;
            ::outputLayer[j]->value = utils::sigmoid(sum);

            test_data[idx].out.push_back(::outputLayer[j]->value);
        }

    }

    for (auto &sample : test_data) {
        for (size_t i = 0; i < INNODE; i++) {
            std::cout << sample.in[i] << " ";
        }
        for (size_t i = 0; i < OUTNODE; i++) {
            std::cout << sample.out[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
