#ifndef EMPIRIC_H
#define EMPIRIC_H

#include "Primary.h"
#include "Mixture.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

class Empiric {
private:
    int n;              // объем выборки
    int k;              // количество интервалов
    double* data;       // массив данных
    double* fr;         // массив относительных частот
    double min_val;     // минимальное значение выборки
    double max_val;     // максимальное значение выборки

public:
    Empiric();
    // Конструкторы
    Empiric(int n0, const Primary& prim, int k0 = 0);
    Empiric(int n0, const Mixture& mixt, int k0 = 0);
    Empiric(int n0, const Empiric& emp, int k0 = 0);

    // Конструктор копирования
    Empiric(const Empiric& other);

    // Оператор присваивания
    Empiric& operator=(const Empiric& other);

    // Деструктор
    ~Empiric();

    // Основные функции распределения
    double pdf(double x) const;
    void moments(double* mean, double* variance, double* skewness, double* kurtosis) const;
    double randNum() const;

    // Функции персистентности
    void save(std::ostream& out) const;
    void load(std::istream& in);

    // Get-функции
    int getN() const { return n; }
    int getK() const { return k; }
    double getMin() const { return min_val; }
    double getMax() const { return max_val; }

private:
    void init_empirical(double* sample_data, int sample_size, int k0);
    void free_memory();
    void copy_data(const Empiric& other);
};

#endif