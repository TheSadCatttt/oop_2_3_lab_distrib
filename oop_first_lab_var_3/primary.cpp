#include "Primary.h"
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Файловый глобальный сид для объектов Primary (0 — означает использовать random_device)
static unsigned int primary_seed = 0;


// Конструктор с параметрами по умолчанию
Primary::Primary(double form, double scale, double shift)
    : mu(shift),
      lambda(scale > 0 ? scale : throw std::invalid_argument("Scale must be positive")),
      v(form > 0 ? form : throw std::invalid_argument("Form parameter must be positive")),
      gen(primary_seed ? primary_seed : std::random_device{}()), unif(0.0, 1.0) {
}

// Конструктор из потока
Primary::Primary(std::istream& in)
    : gen(primary_seed ? primary_seed : std::random_device{}()), unif(0.0, 1.0) {
    load(in);
}

// Установка глобального сида для всех будущих объектов Primary
void Primary::setGlobalSeed(unsigned int seed) {
    primary_seed = seed;
}


    // Set-функции
void Primary::setShift(double newShift) {
    mu = newShift;
}

void Primary::setScale(double newScale) {
    if (newScale <= 0) {
        throw std::invalid_argument("Scale parameter must be positive");
    }
    lambda = newScale;
}

void Primary::setForm(double newForm) {
    if (newForm <= 0) {
        throw std::invalid_argument("Form parameter must be positive");
    }
    v = newForm;
}

// Get-функции
double Primary::getShift() const {
    return mu;
}

double Primary::getScale() const {
    return lambda;
}

double Primary::getForm() const {
    return v;
}

// Плотность стандартного SEN распределения
double Primary::pdf_sen_standard(double x, double v) const {
    if (v <= 0) return 0.0;

    double term = x * x / 2.0 + v;
    double sqrt_term = sqrt(term);

    double numerator = v * exp(v);
    double denominator = 2.0 * sqrt(2.0) * pow(term, 1.5);

    double erfc_term = erfc(sqrt_term);
    double exp_term = (2.0 / sqrt(M_PI)) * sqrt_term * exp(-term);

    double bracket = erfc_term + exp_term;

    return (numerator / denominator) * bracket;
}

// Основная функция плотности
double Primary::pdf(double x) const {
    if (lambda <= 0) return 0.0;
    double z = (x - mu) / lambda;
    return pdf_sen_standard(z, v) / std::abs(lambda);
}

// Моменты распределения
void Primary::moments(double* mean, double* variance, double* skewness, double* kurtosis) const {
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

// Генерация стандартной SEN величины
double Primary::generate_sen_standard(double v) const {
    // Шаг 1: Генерация стандартной нормальной величины (Бокс-Мюллер)
    double r1, r2;
    do {
        r1 = unif(gen);
        r2 = unif(gen);
    } while (r1 == 0.0 || r2 == 0.0);

    double z = sqrt(-2.0 * log(r1)) * cos(2.0 * M_PI * r2);

    // Шаг 2: Генерация случайной величины τ
    double r0;
    do {
        r0 = unif(gen);
    } while (r0 == 0.0);

    double tau = 1.0 - (1.0 / v) * log(r0);

    // SEN стандартная величина
    return z / sqrt(tau);
}

// Генерация случайной величины
double Primary::randNum() const {
    double sen_standard = generate_sen_standard(v);
    return mu + lambda * sen_standard;
}

// Сохранение в поток
void Primary::save(std::ostream& out) const {
    out << mu << " " << lambda << " " << v;
}

// Загрузка из потока
void Primary::load(std::istream& in) {
    in >> mu >> lambda >> v;
    if (lambda <= 0 || v <= 0) {
        throw std::invalid_argument("Invalid parameters in file");
    }
}