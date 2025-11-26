#ifndef MIXTURE_H
#define MIXTURE_H

#include "Primary.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

class Mixture {
private:
    double p;           // параметр смеси
    Primary d1, d2;     // два основных распределения (композиция)

public:
    // Конструкторы
    Mixture(const Primary& prim1, const Primary& prim2, double p0);
    Mixture(std::istream& in);

    // Функции доступа к компонентам
    Primary& component1() { return d1; }
    Primary& component2() { return d2; }
    const Primary& component1() const { return d1; }
    const Primary& component2() const { return d2; }

    // Get-функции
    double getP() const { return p; }

    // Set-функции
    void setP(double newP);

    // Основные функции распределения
    double pdf(double x) const;
    void moments(double* mean, double* variance, double* skewness, double* kurtosis) const;
    double randNum() const;

    // Функции персистентности
    void save(std::ostream& out) const;
    void load(std::istream& in);
};

#endif