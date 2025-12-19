#include "distributions.h"
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

static unsigned int main_seed = 0;
static unsigned int mixture_seed = 0;
static unsigned int empirical_seed = 0;

void set_main_seed(unsigned int seed) { main_seed = seed; }
void set_mixture_seed(unsigned int seed) { mixture_seed = seed; }
void set_empirical_seed(unsigned int seed) { empirical_seed = seed; }

// Вспомогательная функция: дополнительная функция ошибок
double erfcx(double x) {
    return exp(x * x) * erfc(x);
}

// STANDARD SEN PDF (исправленная формула)
double pdf_sen_standard(double x, double v) {
    if (v <= 0) return 0.0;

    double term = x * x / 2.0 + v;
    double sqrt_term = sqrt(term);

    // ИСПРАВЛЕННАЯ ФОРМУЛА согласно методичке
    double numerator = v * exp(v);
    double denominator = 2.0 * sqrt(2.0) * pow(term, 1.5); // Убрано M_PI

    double erfc_term = erfc(sqrt_term);
    double exp_term = (2.0 / sqrt(M_PI)) * sqrt_term * exp(-term);

    double bracket = erfc_term + exp_term;

    return (numerator / denominator) * bracket;
}

// --- MAIN SEN DISTRIBUTION ---
double pdf_main(double x, double mu, double lambda, double v) {
    if (lambda <= 0) return 0.0;

    double z = (x - mu) / lambda;
    return pdf_sen_standard(z, v) / fabs(lambda); // Добавлен fabs для безопасности
}

void moments_main(double mu, double lambda, double v, double* mean,
    double* variance, double* skewness, double* kurtosis) {

    // Табличные значения из методички для v
    double base_variance, base_kurtosis;

    if (v == 0.1) { base_variance = 0.201; base_kurtosis = 2.902; }
    else if (v == 0.25) { base_variance = 0.335; base_kurtosis = 1.437; }
    else if (v == 0.5) { base_variance = 0.461; base_kurtosis = 0.794; }
    else if (v == 1.0) { base_variance = 0.596; base_kurtosis = 0.405; }
    else if (v == 2.0) { base_variance = 0.723; base_kurtosis = 0.186; }
    else if (v == 3.0) { base_variance = 0.786; base_kurtosis = 0.112; }
    else if (v == 4.0) { base_variance = 0.825; base_kurtosis = 0.076; }
    else if (v == 5.0) { base_variance = 0.852; base_kurtosis = 0.055; }
    else if (v == 10.0) { base_variance = 0.916; base_kurtosis = 0.019; }
    *mean = mu;
    *variance = lambda * lambda * base_variance;
    *skewness = 0.0;  // SEN distribution is symmetric
    *kurtosis = base_kurtosis;
}

// GENERATOR
double generate_main(double mu, double lambda, double v) {
    static random_device rd;
    static mt19937 gen(main_seed ? main_seed : random_device{}());
    static uniform_real_distribution<double> unif(0.0, 1.0);
// Бокс-Мюллер
    double r1, r2;
    do {
        r1 = unif(gen);
        r2 = unif(gen);
    } while (r1 == 0.0 || r2 == 0.0);

    // ТОЧНО ПРАВИЛЬНАЯ!
    double z = sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);

    // Генерация случайной величины (экспоненциальная со сдвигом 1)
    double r0;
    do {
        r0 = unif(gen);
    } while (r0 == 0.0);

    // Формула с методички
    double tau = 1.0 - (1.0 / v) * log(r0);

    // SEN стандартная величина
    double sen_standard = z / sqrt(tau);

    //Сдвиг-масштабное преобразование
    return mu + lambda * sen_standard;
}
// Смесь
double pdf_mixture(double x, MixtureParams* params) {
    double pdf1 = pdf_main(x, params->mu1, params->lambda1, params->v1);
    double pdf2 = pdf_main(x, params->mu2, params->lambda2, params->v2);
    return (1.0 - params->p) * pdf1 + params->p * pdf2;
}

void moments_mixture(MixtureParams* params, double* mean, double* variance,
    double* skewness, double* kurtosis) {

    double mean1, var1, skew1, kurt1;
    double mean2, var2, skew2, kurt2;

    moments_main(params->mu1, params->lambda1, params->v1, &mean1, &var1, &skew1, &kurt1);
    moments_main(params->mu2, params->lambda2, params->v2, &mean2, &var2, &skew2, &kurt2);

    // Mean of mixture (формула из методички)
    *mean = (1.0 - params->p) * mean1 + params->p * mean2;

    // Variance of mixture
    *variance = (1.0 - params->p) * (var1 + mean1 * mean1) +
        params->p * (var2 + mean2 * mean2) - (*mean) * (*mean);

    // Skewness of mixture
    double m3_1 = skew1 * pow(var1, 1.5) + 3.0 * mean1 * var1 + pow(mean1, 3);
    double m3_2 = skew2 * pow(var2, 1.5) + 3.0 * mean2 * var2 + pow(mean2, 3);

    double m3_mix = (1.0 - params->p) * m3_1 + params->p * m3_2;
    *skewness = (m3_mix - 3.0 * (*mean) * (*variance) - pow(*mean, 3)) / pow(*variance, 1.5);

    // Kurtosis of mixture
    double m4_1 = (kurt1 + 3.0) * var1 * var1 + 4.0 * skew1 * mean1 * pow(var1, 1.5) +
        6.0 * mean1 * mean1 * var1 + pow(mean1, 4);
    double m4_2 = (kurt2 + 3.0) * var2 * var2 + 4.0 * skew2 * mean2 * pow(var2, 1.5) +
        6.0 * mean2 * mean2 * var2 + pow(mean2, 4);

    double m4_mix = (1.0 - params->p) * m4_1 + params->p * m4_2;
    *kurtosis = (m4_mix - 4.0 * (*mean) * m3_mix + 6.0 * (*mean) * (*mean) * (*variance + (*mean) * (*mean)) -
        3.0 * pow(*mean, 4)) / (*variance * *variance) - 3.0;
}

double generate_mixture(MixtureParams* params) {
    static random_device rd;
    static mt19937 gen(mixture_seed ? mixture_seed : random_device{}());
    static uniform_real_distribution<double> unif(0.0, 1.0);

    if (unif(gen) < params->p) {
        return generate_main(params->mu2, params->lambda2, params->v2);
    }
    else {
        return generate_main(params->mu1, params->lambda1, params->v1);
    }
}

// Эмпирическое
void init_empirical(EmpiricalParams* params, double* data, int n, int k) {
    params->data = new double[n];
    copy(data, data + n, params->data);
    params->n = n;

    // Сортировка данных
    sort(params->data, params->data + n);

    params->min_val = params->data[0];
    params->max_val = params->data[n - 1];

    // Формула Стёрджеса для выбора числа интервалов
    if (k <= 0) {
        k = (int)ceil(1 + log2(n));
    }
    params->k = k;

    // Инициализация частот
    params->frequencies = new double[k];
    double bin_width = (params->max_val - params->min_val) / k;

    // Подсчет частот
    fill(params->frequencies, params->frequencies + k, 0.0);
    for (int i = 0; i < n; i++) {
        int bin_index = (int)((params->data[i] - params->min_val) / bin_width);
        if (bin_index == k) bin_index = k - 1; // Последний элемент попадает в последний bin
        params->frequencies[bin_index] += 1.0;
    }

    // Нормализация частот
    for (int i = 0; i < k; i++) {
        params->frequencies[i] /= (n * bin_width);
    }
}

double pdf_empirical(double x, EmpiricalParams* params) {
    if (x < params->min_val || x > params->max_val) return 0.0;

    double bin_width = (params->max_val - params->min_val) / params->k;
    int bin_index = (int)((x - params->min_val) / bin_width);

    if (bin_index == params->k) bin_index = params->k - 1;

    return params->frequencies[bin_index];
}

void moments_empirical(EmpiricalParams* params, double* mean, double* variance,
    double* skewness, double* kurtosis) {

    // Выборочное среднее
    *mean = accumulate(params->data, params->data + params->n, 0.0) / params->n;

    // Выборочная дисперсия
    double sum_sq = 0.0, sum_cube = 0.0, sum_quad = 0.0;
    for (int i = 0; i < params->n; i++) {
        double dev = params->data[i] - *mean;
        double dev_sq = dev * dev;
        sum_sq += dev_sq;
        sum_cube += dev_sq * dev;
        sum_quad += dev_sq * dev_sq;
    }

    *variance = sum_sq / (params->n - 1);
    double std_dev = sqrt(*variance);

    // Коэффициент асимметрии
    *skewness = (sum_cube / params->n) / pow(*variance, 1.5);

    // Коэффициент эксцесса
    *kurtosis = (sum_quad / params->n) / (*variance * *variance) - 3.0;
}

double generate_empirical(EmpiricalParams* params) {
    static random_device rd;
    static mt19937 gen(empirical_seed ? empirical_seed : random_device{}());
    static uniform_real_distribution<double> unif(0.0, 1.0);

    // Метод суперпозиции для эмпирического распределения
    double bin_width = (params->max_val - params->min_val) / params->k;

    // Выбор бина согласно относительным частотам
    double r = unif(gen);
    double cumulative = 0.0;
    int selected_bin = 0;

    for (int i = 0; i < params->k; i++) {
        double prob = params->frequencies[i] * bin_width;
        cumulative += prob;
        if (r <= cumulative) {
            selected_bin = i;
            break;
        }
    }

    // Генерация равномерно в выбранном бине
    double left = params->min_val + selected_bin * bin_width;
    double right = left + bin_width;

    return left + unif(gen) * bin_width;
}

void free_empirical(EmpiricalParams* params) {
    delete[] params->data;
    delete[] params->frequencies;
    params->n = 0;
    params->k = 0;
}