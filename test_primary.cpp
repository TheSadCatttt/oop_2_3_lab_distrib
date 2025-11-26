#include "Primary.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void test_primary_class() {
    std::cout << "\n=== Тестирование класса Primary ===" << std::endl;

    try {
        // Тест 1: Создание объектов
        std::cout << "1. Создание объектов:" << std::endl;
        Primary prim1; // по умолчанию
        Primary prim2(1.0, 2.0, 0.0); // с параметрами
        Primary prim3(0.5, 1.0, 1.0);

        std::cout << "   prim1: v=" << prim1.getForm() << ", λ=" << prim1.getScale()
            << ", μ=" << prim1.getShift() << std::endl;
        std::cout << "   prim2: v=" << prim2.getForm() << ", λ=" << prim2.getScale()
            << ", μ=" << prim2.getShift() << std::endl;

        // Тест 2: Set-функции
        std::cout << "2. Тестирование set-функций:" << std::endl;
        prim1.setShift(2.0);
        prim1.setScale(1.5);
        prim1.setForm(2.0);
        std::cout << "   После изменения: v=" << prim1.getForm() << ", λ=" << prim1.getScale()
            << ", μ=" << prim1.getShift() << std::endl;

        // Тест 3: Плотность распределения
        std::cout << "3. Тестирование плотности:" << std::endl;
        Primary testPdf(1.0, 1.0, 0.0); // SEN(0,1,1)
        double pdf_val = testPdf.pdf(0.0);
        std::cout << "   f(0) для SEN(0,1,1) = " << pdf_val << " (ожидалось ~0.55)" << std::endl;

        // Тест 4: Моменты распределения
        std::cout << "4. Тестирование моментов:" << std::endl;
        double mean, variance, skewness, kurtosis;
        testPdf.moments(&mean, &variance, &skewness, &kurtosis);
        std::cout << "   Mξ = " << mean << " (ожидалось 0.0)" << std::endl;
        std::cout << "   Dξ = " << variance << " (ожидалось 0.596)" << std::endl;
        std::cout << "   γ₁ = " << skewness << " (ожидалось 0.0)" << std::endl;
        std::cout << "   γ₂ = " << kurtosis << " (ожидалось 0.405)" << std::endl;

        // Тест 5: Генерация случайных величин
        std::cout << "5. Тестирование генерации:" << std::endl;
        double sum = 0.0;
        int n = 1000;
        for (int i = 0; i < n; i++) {
            sum += testPdf.randNum();
        }
        std::cout << "   Среднее " << n << " величин = " << sum / n << " (ожидалось ~0.0)" << std::endl;

        // Тест 6: Персистентность
        std::cout << "6. Тестирование персистентности:" << std::endl;
        std::ofstream outFile("primary_test.txt");
        if (outFile) {
            prim1.save(outFile);
            outFile.close();
            std::cout << "   Объект сохранен в файл" << std::endl;
        }

        std::ifstream inFile("primary_test.txt");
        if (inFile) {
            Primary primFromFile(inFile);
            inFile.close();
            std::cout << "   Объект загружен из файла: v=" << primFromFile.getForm()
                << ", λ=" << primFromFile.getScale() << ", μ=" << primFromFile.getShift() << std::endl;
        }

        // Тест 7: Обработка исключений
        std::cout << "7. Тестирование обработки исключений:" << std::endl;
        try {
            Primary invalid(0.0, 1.0, 0.0); // недопустимый параметр формы
        }
        catch (const std::exception& e) {
            std::cout << "   Перехвачено исключение: " << e.what() << std::endl;
        }

        std::cout << "✓ Все тесты пройдены успешно!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }
}

void demonstrate_primary_usage() {
    std::cout << "\n=== Демонстрация использования класса Primary ===" << std::endl;

    // Создание распределений для тестов 3.1.1-3.1.3
    Primary std_dist(1.0, 1.0, 0.0);      // 3.1.1: SEN(0,1,1)
    Primary scaled_dist(1.0, 2.0, 0.0);   // 3.1.2: SEN(0,2,1)  
    Primary shift_scaled_dist(1.0, 2.0, 5.0); // 3.1.3: SEN(5,2,1)

    std::cout << "Созданные распределения:" << std::endl;
    std::cout << "  SEN(0,1,1):  μ=" << std_dist.getShift() << ", λ=" << std_dist.getScale()
        << ", v=" << std_dist.getForm() << std::endl;
    std::cout << "  SEN(0,2,1):  μ=" << scaled_dist.getShift() << ", λ=" << scaled_dist.getScale()
        << ", v=" << scaled_dist.getForm() << std::endl;
    std::cout << "  SEN(5,2,1): μ=" << shift_scaled_dist.getShift() << ", λ=" << shift_scaled_dist.getScale()
        << ", v=" << shift_scaled_dist.getForm() << std::endl;

    // Вычисление плотностей в нескольких точках
    std::cout << "\nЗначения плотностей:" << std::endl;
    double points[] = { -2.0, -1.0, 0.0, 1.0, 2.0 };
    for (double x : points) {
        std::cout << "  x=" << std::setw(4) << x
            << "  f_std=" << std::setw(8) << std_dist.pdf(x)
            << "  f_scaled=" << std::setw(8) << scaled_dist.pdf(x)
            << "  f_shift=" << std::setw(8) << shift_scaled_dist.pdf(x) << std::endl;
    }

    // Генерация выборки
    std::cout << "\nГенерация выборки из SEN(0,1,1):" << std::endl;
    std::cout << "  ";
    for (int i = 0; i < 5; i++) {
        std::cout << std_dist.randNum() << " ";
    }
    std::cout << "...";
    std::cout << std::endl;
    test_primary_class();
    demonstrate_primary_usage();
}

/*
int main() {
    std::cout << "ТЕСТИРОВАНИЕ КЛАССА PRIMARY ДЛЯ SEN-РАСПРЕДЕЛЕНИЯ" << std::endl;
    std::cout << "==================================================" << std::endl;

    test_primary_class();
    demonstrate_primary_usage();

    return 0;
}*/