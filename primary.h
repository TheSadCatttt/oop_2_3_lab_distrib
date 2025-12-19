#ifndef PRIMARY_H
#define PRIMARY_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

class Primary {
private:
    double mu;       //сдвиг    
	double lambda;  //масштаб
    double v;       //форма

	//генерация случайных чисел
    mutable std::mt19937 gen;
    mutable std::uniform_real_distribution<double> unif;

public:

    // Глобальная установка сида для всех объектов Primary
    static void setGlobalSeed(unsigned int seed);


    Primary(double form = 1.0, double scale = 1.0, double shift = 0.0);
    Primary(std::istream& in);

	// Set-методы
    void setShift(double newShift);
    void setScale(double newScale);
    void setForm(double newForm);

    // Get-методы
    double getShift() const;
    double getScale() const;
    double getForm() const;

	// аналитические функции
    double pdf(double x) const;
    void moments(double* mean, double* variance, double* skewness, double* kurtosis) const;
    double randNum() const;

	// функции персистентности
    void save(std::ostream& out) const;
    void load(std::istream& in);

	// вспомогательные функции
private:
    double pdf_sen_standard(double x, double v) const;
    double generate_sen_standard(double v) const;
};

#endif