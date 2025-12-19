#include "Mixture.h"
#include <stdexcept>

// Конструктор
Mixture::Mixture(const Primary& prim1, const Primary& prim2, double p0) //из праймари
    : p(p0 > 0 && p0 < 1 ? p0 : throw std::invalid_argument("Parameter p must be in (0,1)")),
    d1(prim1), d2(prim2) {
}

// Конструктор из потока
Mixture::Mixture(std::istream& in) {
    load(in);
}

// Set-функция для параметра смеси
void Mixture::setP(double newP) {
    if (newP <= 0 || newP >= 1) {
        throw std::invalid_argument("Parameter p must be in (0,1)");
    }
    p = newP;
}

// Плотность смеси
double Mixture::pdf(double x) const {
    return (1.0 - p) * d1.pdf(x) + p * d2.pdf(x);
}

// Моменты смеси
void Mixture::moments(double* mean, double* variance, double* skewness, double* kurtosis) const {
    double mean1, var1, skew1, kurt1;
    double mean2, var2, skew2, kurt2;

    d1.moments(&mean1, &var1, &skew1, &kurt1);
    d2.moments(&mean2, &var2, &skew2, &kurt2);

    // Mean of mixture
    *mean = (1.0 - p) * mean1 + p * mean2;

    // Variance of mixture
    *variance = (1.0 - p) * (var1 + mean1 * mean1) +
        p * (var2 + mean2 * mean2) - (*mean) * (*mean);

    // Skewness of mixture
    double m3_1 = skew1 * pow(var1, 1.5) + 3.0 * mean1 * var1 + pow(mean1, 3);
    double m3_2 = skew2 * pow(var2, 1.5) + 3.0 * mean2 * var2 + pow(mean2, 3);

    double m3_mix = (1.0 - p) * m3_1 + p * m3_2;
    *skewness = (m3_mix - 3.0 * (*mean) * (*variance) - pow(*mean, 3)) / pow(*variance, 1.5);

    // Kurtosis of mixture
    double m4_1 = (kurt1 + 3.0) * var1 * var1 + 4.0 * skew1 * mean1 * pow(var1, 1.5) +
        6.0 * mean1 * mean1 * var1 + pow(mean1, 4);
    double m4_2 = (kurt2 + 3.0) * var2 * var2 + 4.0 * skew2 * mean2 * pow(var2, 1.5) +
        6.0 * mean2 * mean2 * var2 + pow(mean2, 4);

    double m4_mix = (1.0 - p) * m4_1 + p * m4_2;
    *kurtosis = (m4_mix - 4.0 * (*mean) * m3_mix + 6.0 * (*mean) * (*mean) * (*variance + (*mean) * (*mean)) -
        3.0 * pow(*mean, 4)) / (*variance * *variance) - 3.0;
}

// Генерация случайной величины
double Mixture::randNum() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> unif(0.0, 1.0);

    if (unif(gen) < p) {
        return d2.randNum();
    }
    else {
        return d1.randNum();
    }
}

// Сохранение в поток
void Mixture::save(std::ostream& out) const {
    out << p << " ";
    d1.save(out);
    out << " ";
    d2.save(out);
}

// Загрузка из потока
void Mixture::load(std::istream& in) {
    in >> p;
    d1.load(in);
    d2.load(in);

    if (p <= 0 || p >= 1) {
        throw std::invalid_argument("Invalid parameter p in file");
    }
}