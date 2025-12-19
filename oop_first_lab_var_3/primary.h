#ifndef PRIMARY_H
#define PRIMARY_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

class Primary {
private:
    // Обязательные атрибуты
    double mu;      // параметр сдвига
    double lambda;  // параметр масштаба
    double v;       // параметр формы

    // Вспомогательные атрибуты для эффективности
    mutable std::mt19937 gen;
    mutable std::uniform_real_distribution<double> unif;

public:
    // Конструкторы
    Primary(double form = 1.0, double scale = 1.0, double shift = 0.0);
    Primary(std::istream& in);

    // Глобальная установка сида для всех объектов Primary
    static void setGlobalSeed(unsigned int seed);


    // Set-функции
    void setShift(double newShift);
    void setScale(double newScale);
    void setForm(double newForm);

    // Get-функции
    double getShift() const;
    double getScale() const;
    double getForm() const;

    // Основные функции распределения
    double pdf(double x) const;
    void moments(double* mean, double* variance, double* skewness, double* kurtosis) const;
    double randNum() const;

    // Функции персистентности
    void save(std::ostream& out) const;
    void load(std::istream& in);

    // Вспомогательные функции
private:
    double pdf_sen_standard(double x, double v) const;
    double generate_sen_standard(double v) const;
};

#endif