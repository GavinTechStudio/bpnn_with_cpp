#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc51-cpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>

#define INNODE 2
#define HIDDENNODE 4
#define OUTNODE 1
#define RATE 0.9

double threshold = 1e-4;
int mosttimes = 1e6;

class Sample {
public:
    std::vector<double> in, out;
};

class Node {
public:
    double value{}, bias{}, bias_delta{};
    std::vector<double> weight, weight_delta;
};


namespace utils {
    inline double getRandom() {
        return (2.0 * (double) std::rand() / RAND_MAX) - 1.0;
    }

    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::vector<double> getFileData(char *filename) {
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

    std::vector<Sample> getTrainData(char *filename) {
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

    std::vector<Sample> getTestData(char *filename) {
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
    std::srand((unsigned) std::time(NULL));

    for (int i = 0; i < INNODE; i++) {
        ::inputLayer[i] = new Node();
        for (int j = 0; j < HIDDENNODE; j++) {
            ::inputLayer[i]->weight.push_back(utils::getRandom());
            ::inputLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (int i = 0; i < HIDDENNODE; i++) {
        ::hiddenLayer[i] = new Node();
        ::hiddenLayer[i]->bias = utils::getRandom();
        for (int j = 0; j < OUTNODE; j++) {
            ::hiddenLayer[i]->weight.push_back(utils::getRandom());
            ::hiddenLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (int i = 0; i < OUTNODE; i++) {
        ::outputLayer[i] = new Node();
        ::outputLayer[i]->bias = utils::getRandom();
    }
}

inline void reset_delta() {
    for (int i = 0; i < INNODE; i++) {
        ::inputLayer[i]->weight_delta.assign(::inputLayer[i]->weight_delta.size(), 0.f);
    }
    for (int i = 0; i < HIDDENNODE; i++) {
        ::hiddenLayer[i]->bias_delta = 0.f;
        ::hiddenLayer[i]->weight_delta.assign(::hiddenLayer[i]->weight_delta.size(), 0.f);
    }
    for (int i = 0; i < OUTNODE; i++) {
        ::outputLayer[i]->bias_delta = 0.f;
    }
}

int main() {

    init();

    std::vector<Sample> train_data = utils::getTrainData("traindata.txt");

    for (int times = 0; times < mosttimes; times++) {

        reset_delta();

        double error_max = 0;

        for (size_t idx = 0; idx < train_data.size(); idx++) {

            for (int i = 0; i < INNODE; i++) {
                ::inputLayer[i]->value = train_data[idx].in[i];
            }

            // 正向传播
            for (int j = 0; j < HIDDENNODE; j++) {
                double sum = 0;
                for (int i = 0; i < INNODE; i++) {
                    sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
                }
                sum -= ::hiddenLayer[j]->bias;
                ::hiddenLayer[j]->value = utils::sigmoid(sum);
            }

            for (int j = 0; j < OUTNODE; j++) {
                double sum = 0;
                for (int i = 0; i < HIDDENNODE; i++) {
                    sum += ::hiddenLayer[i]->value * ::hiddenLayer[i]->weight[j];
                }
                sum -= ::outputLayer[j]->bias;
                ::outputLayer[j]->value = utils::sigmoid(sum);
            }

            // 计算损失函数
            double error = 0;
            for (int i = 0; i < OUTNODE; i++) {
                double tmp = std::fabs(::outputLayer[i]->value - train_data[idx].out[i]);
                error += tmp * tmp / 2;
            }

            error_max = std::max(error_max, error);

            // 反向传播
            for (int i = 0; i < OUTNODE; i++) {
                double bias_delta = -(train_data[idx].out[i] - ::outputLayer[i]->value) *
                        ::outputLayer[i]->value * (1.0 - ::outputLayer[i]->value);
                ::outputLayer[i]->bias_delta += bias_delta;
            }

            for (int i = 0; i < HIDDENNODE; i++) {
                for (int j = 0; j < OUTNODE; j++) {
                    double weight_delta = (train_data[idx].out[j] - ::outputLayer[j]->value) *
                            ::outputLayer[j]->value * (1.0 - ::outputLayer[j]->value) *
                            ::hiddenLayer[i]->value;
                    ::hiddenLayer[i]->weight_delta[j] += weight_delta;
                }
            }

            for (int i = 0; i < HIDDENNODE; i++) {
                double sum = 0;
                for (int j = 0; j < OUTNODE; j++) {
                    sum += -(train_data[idx].out[j] - ::outputLayer[j]->value) *
                           ::outputLayer[j]->value * (1.0 - ::outputLayer[j]->value) *
                           ::hiddenLayer[i]->weight[j];
                }
                ::hiddenLayer[i]->bias_delta +=
                        sum * ::hiddenLayer[i]->value * (1.0 - ::hiddenLayer[i]->value);
            }

            for (int i = 0; i < INNODE; i++) {
                for (int j = 0; j < HIDDENNODE; j++) {
                    double sum = 0;
                    for (int k = 0; k < OUTNODE; k++) {
                        sum += (train_data[idx].out[k] - ::outputLayer[k]->value) *
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

        for (int i = 0; i < INNODE; i++) {
            for (int j = 0; j < HIDDENNODE; j++) {
                ::inputLayer[i]->weight[j] += RATE * ::inputLayer[i]->weight_delta[j] / train_data.size();
            }
        }

        for (int i = 0; i < HIDDENNODE; i++) {
            ::hiddenLayer[i]->bias += RATE * ::hiddenLayer[i]->bias_delta / train_data.size();
            for (int j = 0; j < OUTNODE; j++) {
                ::hiddenLayer[i]->weight[j] += RATE * ::hiddenLayer[i]->weight_delta[j] / train_data.size();
            }
        }

        for (int i = 0; i < OUTNODE; i++) {
            ::outputLayer[i]->bias += RATE * ::outputLayer[i]->bias_delta / train_data.size();
        }

    }

    std::vector<Sample> test_data = utils::getTestData("testdata.txt");

    for (size_t idx = 0; idx < test_data.size(); idx++) {

        for (int i = 0; i < INNODE; i++) {
            inputLayer[i]->value = test_data[idx].in[i];
        }

        for (int j = 0; j < HIDDENNODE; j++) {
            double sum = 0;
            for (int i = 0; i < INNODE; i++) {
                sum += inputLayer[i]->value * inputLayer[i]->weight[j];
            }
            sum -= ::hiddenLayer[j]->bias;
            ::hiddenLayer[j]->value = utils::sigmoid(sum);
        }

        for (int j = 0; j < OUTNODE; j++) {
            double sum = 0;
            for (int i = 0; i < HIDDENNODE; i++) {
                sum += ::hiddenLayer[i]->value * ::hiddenLayer[i]->weight[j];
            }
            sum -= ::outputLayer[j]->bias;
            ::outputLayer[j]->value = utils::sigmoid(sum);

            test_data[idx].out.push_back(::outputLayer[j]->value);
        }

    }

    for (auto &sample : test_data) {
        for (int i = 0; i < INNODE; i++) {
            std::cout << sample.in[i] << " ";
        }
        for (int i = 0; i < OUTNODE; i++) {
            std::cout << sample.out[i] << " ";
        }
        std::cout << std::endl;
    }


    return 0;
}
#pragma clang diagnostic pop
