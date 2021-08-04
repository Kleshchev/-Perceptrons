#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#define MAX_POP 50
#define MAX_GENERATIONS 1000
#define MAX_MUTATE MAX_POP / 2
#define PARENTS_COUNT MAX_POP / 10
#define WEIGHTS_RATE 10
#define MAX_MUTATE_RATE 2 * WEIGHTS_RATE / 5
#define RELEARN_RATE 1

using namespace std;

float sigmoid(float x) { return 1 / (1 + exp(-x)); }
int data_in[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
int answers[4]    = {0, 1, 1, 0};
class XorIns {
   public:
	float bias_out;
	float error_sum = 0;
	vector<float> output;
	vector<vector<float>> weights_in;
	vector<float> weights_out;
	vector<float> bias_in;
	vector<vector<float>> input;
	vector<float> hidden_layer;
	vector<float> errors;
	float randf(float min, float max) {
		return (max - min) * (float)rand() / (float)RAND_MAX + min;
	}
	XorIns() {
		input        = vector<vector<float>>(4, vector<float>(2));
		output       = vector<float>(4);
		hidden_layer = vector<float>(3);
		errors       = vector<float>(4);
		generate_weights();
		for (int i = 0; i < 4; i++) {
			input[i][0] = data_in[i][0];
			input[i][1] = data_in[i][1];
		}
	}
	void set_this(XorIns t) {
		XorIns();
		set_bias_in(t.bias_in);
		set_bias_out(t.bias_out);
		set_in_w(t.weights_in);
		set_out_w(t.weights_out);
		calculate_errors();
	}

	string w_to_string() {
		string res = "InWeights:\n";
		for (int i = 0; i < 3; i++) {
			res += "in0: " + to_string(weights_in[i][0]) +
			       " n1: " + to_string(weights_in[i][1]) + " bias_in " +
			       to_string(bias_in[i]) + " h " + to_string(weights_out[i]) +
			       " bias_out " + to_string(bias_out) + "\n";
		}
		res += to_string(error_sum) + "\n";
		return res;
	}
	void generate_weights() {
		weights_in  = vector<vector<float>>(3);
		weights_out = vector<float>(3);
		bias_in     = vector<float>(3);
		for (int i = 0; i < 3; i++) {
			vector<float> t;
			t.push_back(randf(-WEIGHTS_RATE, WEIGHTS_RATE));
			t.push_back(randf(-WEIGHTS_RATE, WEIGHTS_RATE));
			weights_in[i]  = t;
			weights_out[i] = randf(-WEIGHTS_RATE, WEIGHTS_RATE);
			bias_out       = randf(-WEIGHTS_RATE, WEIGHTS_RATE);
			bias_in[i]     = randf(-WEIGHTS_RATE, WEIGHTS_RATE);
		}
	}

	void set_in_w(vector<vector<float>> &in) { weights_in = in; }
	vector<vector<float>> get_in_w() { return weights_in; }
	void set_out_w(vector<float> out) { weights_out = out; }
	vector<float> get_out_w() { return weights_out; }
	void set_bias_in(vector<float> bias) { bias_in = bias; }
	vector<float> get_bias_in() { return bias_in; }
	void set_bias_out(float bias) { bias_out = bias; }
	float get_bias_out() { return bias_out; }
	void calculate_output() {
		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 3; i++) {
				hidden_layer[i] =
				    sigmoid(input[j][0] * weights_in[i][0] +
				            input[j][1] * weights_in[i][1] + bias_in[i]);
			}
			float res = 0;
			for (int i = 0; i < 3; i++) {
				res += hidden_layer[i] * weights_out[i] + bias_out;
			}
			output[j] = sigmoid(res);
		}
	}
	void calculate_errors() {
		for (int i = 0; i < 4; i++) {
			calculate_output();
			errors[i] = answers[i] - output[i];
		}

		error_sum = 0;
		for (auto er : errors) {
			error_sum += abs(er);
		}
	}
	float get_delta_w() { return randf(-MAX_MUTATE_RATE, MAX_MUTATE_RATE); }
	void mutate() {
		for (auto w_in : weights_in) {
			for (int i = 0; i < rand() % w_in.size(); i++) {
				w_in[rand() % w_in.size()] += get_delta_w();
			}
		}
		for (int i = 0; i < rand() % weights_out.size(); i++) {
			weights_out[rand() % weights_out.size()] += get_delta_w();
		}
		for (int i = 0; i < rand() % bias_in.size(); i++) {
			bias_in[rand() % bias_in.size()] += get_delta_w();
		}
		bias_out += get_delta_w();
	}
};
class GA {
   public:
	vector<XorIns> population;
	GA() { population = vector<XorIns>(MAX_POP); }
	void sort_population() {
		sort(population.begin(), population.end(),
		     [&](XorIns a, XorIns b) -> bool {
			     return a.error_sum < b.error_sum;
		     });
	}
	// only if population is sorted
	vector<XorIns> get_parents(int count) {
		vector<XorIns> t;
		for (int i = 0; i < count; i++) {
			t.push_back(population[i]);
		}
		return t;
	}
	XorIns breed(XorIns parent1, XorIns parent2) {
		auto w_in  = parent1.get_in_w();
		auto w_out = parent1.get_out_w();
		auto b_in  = parent1.get_bias_in();
		auto b_out = parent1.get_bias_out();
		for (int i = 0; i < w_in.size(); i++) {
			for (int j = 0; j < w_in[i].size(); j++) {
				w_in[i][j] =
				    (rand() % 2 == 0 ? w_in[i][j] : parent2.weights_in[i][j]);
			}
		}
		for (int i = 0; i < w_out.size(); i++) {
			w_out[i] = (rand() % 2 == 0 ? w_out[i] : parent2.weights_out[i]);
		}
		for (int i = 0; i < b_in.size(); i++) {
			b_in[i] = (rand() % 2 == 0 ? b_in[i] : parent2.bias_in[i]);
		}
		b_out = (rand() % 2 == 0 ? b_out : parent2.bias_out);
		parent1.set_bias_in(b_in);
		parent1.set_bias_out(b_out);
		parent1.set_in_w(w_in);
		parent1.set_out_w(w_out);
		parent1.calculate_errors();
		return parent1;
	}
	void calculate_population() {
		for (auto x : population) {
			x.calculate_errors();
		}
	}
	float get_errors_sum() {
		float sum = 0;
		for (auto x : population) {
			sum += x.error_sum;
		}
		return sum;
	}
	void proccess() {
		for (int i = 0; i < MAX_POP; i++) {
			XorIns t;
			population[i].set_this(t);
			population[i].calculate_errors();
		}
		for (int i = 0; i < MAX_GENERATIONS; i++) {
			sort_population();
			vector<XorIns> parents;
			for (int i = 0; i < PARENTS_COUNT; i++) {
				parents.push_back(population[i]);
			}
			population.clear();
			population = vector<XorIns>(MAX_POP);
			for (int i = 0; i < MAX_POP; i++) {
				if (i < parents.size()) {
					population[i] = parents[i];
				} else {
					int index1 = rand() % parents.size();
					int index2 = rand() % parents.size();
					while (index1 == index2) {
						index2 = rand() % parents.size();
					}
					population[i] = breed(parents[index1], parents[index2]);
				}
			}
			for (int i = 0; i < rand() % (MAX_MUTATE); i++) {
				population[rand() % (MAX_POP - parents.size()) + parents.size()]
				    .mutate();
			}
			calculate_population();
			cout << get_errors_sum() << endl;
		}
		if (population[0].error_sum > RELEARN_RATE) {
			proccess();
		} else {
			cout << population[0].w_to_string() << endl;
			for (int i = 0; i < population[i].output.size(); i++) {
				cout <<"["<< round(population[i].output[i]) << "] "
				     <<setprecision(1)<< population[i].output[i] << endl;
			}
		}
	}
};
int main() {
	setlocale(LC_ALL, "Russian");
	srand(0);
	GA ga = GA();
	ga.proccess();
	cin.get();
}
