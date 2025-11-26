#ifndef PRIMARY_H
#define PRIMARY_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

class Primary {
private:
    double mu;      
    double lambda;  // �������� ��������
    double v;       // �������� �����

    // ��������������� �������� ��� �������������
    mutable std::mt19937 gen;
    mutable std::uniform_real_distribution<double> unif;

public:
    // ������������
    Primary(double form = 1.0, double scale = 1.0, double shift = 0.0);
    Primary(std::istream& in);

    // Set-�������
    void setShift(double newShift);
    void setScale(double newScale);
    void setForm(double newForm);

    // Get-�������
    double getShift() const;
    double getScale() const;
    double getForm() const;

    // �������� ������� �������������
    double pdf(double x) const;
    void moments(double* mean, double* variance, double* skewness, double* kurtosis) const;
    double randNum() const;

    // ������� ���������������
    void save(std::ostream& out) const;
    void load(std::istream& in);

    // ��������������� �������
private:
    double pdf_sen_standard(double x, double v) const;
    double generate_sen_standard(double v) const;
};

#endif