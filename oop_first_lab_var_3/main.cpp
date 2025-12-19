#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>
#include "distributions.h"
#include "primary.h"
#include "Mixture.h"
#include "Empiric.h"

using namespace std;

// ГЛОБАЛЬНЫЕ КОНСТАНТЫ
const int DEFAULT_SEN_SAMPLE_SIZE = 10000;
const int DEFAULT_MIXTURE_SAMPLE_SIZE = 10000;
const int EMPIRICAL_SIZES[] = { 100, 1000, 10000 };

// ОБЪЯВЛЕНИЯ ФУНКЦИЙ
void input_sample_sizes(int& sen_size, int& mix_size, int empirical_sizes[]);
void test_emяяpirical_distribution();
void generate_all_plot_data(int sen_size, int mix_size, int empirical_sizes[]);
void generate_all_plot_data();
void test_primary_class_menu();
void test_primary_class();
void run_basic_tests();

// ФУНКЦИЯ ВВОДА ДАННЫХ С КЛАВИАТУРЫ
void input_sample_sizes(int& sen_size, int& mix_size, int empirical_sizes[]) {
    cout << "\n--- Ввод размеров выборок ---" << endl;

    cout << "Размер выборки для основных распределений SEN: ";
    cin >> sen_size;

    cout << "Размер выборки для смесей распределений: ";
    cin >> mix_size;

    cout << "Размеры выборок для эмпирического распределения (3 значения): ";
    cin >> empirical_sizes[0] >> empirical_sizes[1] >> empirical_sizes[2];

    // Проверка корректности ввода
    if (sen_size <= 0) sen_size = DEFAULT_SEN_SAMPLE_SIZE;
    if (mix_size <= 0) mix_size = DEFAULT_MIXTURE_SAMPLE_SIZE;
    if (empirical_sizes[0] <= 0) empirical_sizes[0] = EMPIRICAL_SIZES[0];
    if (empirical_sizes[1] <= 0) empirical_sizes[1] = EMPIRICAL_SIZES[1];
    if (empirical_sizes[2] <= 0) empirical_sizes[2] = EMPIRICAL_SIZES[2];
}

// ТЕСТИРОВАНИЕ ЭМПИРИЧЕСКОГО РАСПРЕДЕЛЕНИЯ
void test_empirical_distribution() {
    cout << "\n=== Тестирование эмпирического распределения ===" << endl;

    int sample_size;
    cout << "Введите размер выборки для тестирования: ";
    cin >> sample_size;

    if (sample_size <= 0) sample_size = 1000;

    cout << "Генерация выборки из SEN(0,1,1) объемом " << sample_size << " элементов..." << endl;
    double* sample = new double[sample_size];
    for (int i = 0; i < sample_size; i++) {
        sample[i] = generate_main(0.0, 1.0, 1.0);
    }

    // Создаем эмпирическое распределение
    EmpiricalParams emp_params;
    int bins;
    cout << "Введите количество интервалов для гистограммы: ";
    cin >> bins;
    if (bins <= 0) bins = 20;

    init_empirical(&emp_params, sample, sample_size, bins);

    // Вычисляем моменты
    double mean, variance, skewness, kurtosis;
    moments_empirical(&emp_params, &mean, &variance, &skewness, &kurtosis);

    cout << "\nЭмпирические моменты:" << endl;
    cout << "  Математическое ожидание = " << mean << " (ожидалось 0.0)" << endl;
    cout << "  Дисперсия = " << variance << " (ожидалось 0.596)" << endl;
    cout << "  Асимметрия = " << skewness << " (ожидалось 0.0)" << endl;
    cout << "  Эксцесс = " << kurtosis << " (ожидалось 0.405)" << endl;

    // Генерируем новую выборку из эмпирического распределения
    cout << "\nГенерация новой выборки из эмпирического распределения..." << endl;
    double* new_sample = new double[sample_size];
    for (int i = 0; i < sample_size; i++) {
        new_sample[i] = generate_empirical(&emp_params);
    }

    // Сравниваем характеристики
    EmpiricalParams new_emp_params;
    init_empirical(&new_emp_params, new_sample, sample_size, bins);

    double new_mean, new_variance, new_skewness, new_kurtosis;
    moments_empirical(&new_emp_params, &new_mean, &new_variance, &new_skewness, &new_kurtosis);

    cout << "\nХарактеристики новой выборки:" << endl;
    cout << "  Математическое ожидание = " << new_mean << endl;
    cout << "  Дисперсия = " << new_variance << endl;
    cout << "  Асимметрия = " << new_skewness << endl;
    cout << "  Эксцесс = " << new_kurtosis << endl;

    // Очистка
    free_empirical(&emp_params);
    free_empirical(&new_emp_params);
    delete[] sample;
    delete[] new_sample;

    cout << "Тестирование завершено!" << endl;
}

// ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ
void generate_all_plot_data(int sen_size, int mix_size, int empirical_sizes[]) {
    cout << "\n=== Генерация данных для построения графиков ===" << endl;

    cout << "Используемые размеры выборок:" << endl;
    cout << "  SEN распределения: " << sen_size << " элементов" << endl;
    cout << "  Смеси распределений: " << mix_size << " элементов" << endl;
    cout << "  Эмпирические распределения: " << empirical_sizes[0] << ", "
        << empirical_sizes[1] << ", " << empirical_sizes[2] << " элементов" << endl;

    // 1. Основные распределения SEN (3.1.1 - 3.1.3)
    cout << "\n1. Генерация данных для основных распределений SEN..." << endl;
    ofstream file_sen_main("data_sen_main.csv");
    file_sen_main << "x,std_01_11,scaled_02_11,shift_scaled_52_21\n";

    for (double x = -5.0; x <= 5.0; x += 0.1) {
        double pdf_std = pdf_main(x, 0.0, 1.0, 1.0);
        double pdf_scaled = pdf_main(x, 0.0, 2.0, 1.0);
        double pdf_shift_scaled = pdf_main(x, 5.0, 2.0, 1.0);

        file_sen_main << x << "," << pdf_std << "," << pdf_scaled << "," << pdf_shift_scaled << "\n";
    }
    file_sen_main.close();
    cout << "   Файл data_sen_main.csv создан" << endl;

    // 2. Эмпирические данные для основных распределений
    cout << "2. Генерация эмпирических данных SEN..." << endl;
    ofstream file_sen_empirical("data_sen_empirical.csv");
    file_sen_empirical << "sample_std,sample_scaled,sample_shift_scaled\n";

    for (int i = 0; i < sen_size; i++) {
        double sample_std = generate_main(0.0, 1.0, 1.0);
        double sample_scaled = generate_main(0.0, 2.0, 1.0);
        double sample_shift_scaled = generate_main(5.0, 2.0, 1.0);

        file_sen_empirical << sample_std << "," << sample_scaled << "," << sample_shift_scaled << "\n";
    }
    file_sen_empirical.close();
    cout << "   Файл data_sen_empirical.csv создан (" << sen_size << " элементов)" << endl;

    // 3. Смеси распределений (3.2.1 - 3.2.4)
    cout << "3. Генерация данных для смесей распределений..." << endl;
    ofstream file_mixture_theoretical("data_mixture_theoretical.csv");
    file_mixture_theoretical << "x,trivial,shift,scale,shape\n";

    MixtureParams trivial = { 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.75 };
    MixtureParams shift = { 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.75 };
    MixtureParams scale_mix = { 0.0, 1.0, 1.0, 0.0, 3.0, 1.0, 0.5 };
    MixtureParams shape = { 0.0, 1.0, 0.5, 0.0, 1.0, 2.0, 0.5 };

    for (double x = -6.0; x <= 6.0; x += 0.12) {
        double pdf_trivial = pdf_mixture(x, &trivial);
        double pdf_shift = pdf_mixture(x, &shift);
        double pdf_scale = pdf_mixture(x, &scale_mix);
        double pdf_shape = pdf_mixture(x, &shape);

        file_mixture_theoretical << x << "," << pdf_trivial << "," << pdf_shift << "," << pdf_scale << "," << pdf_shape << "\n";
    }
    file_mixture_theoretical.close();
    cout << "   Файл data_mixture_theoretical.csv создан" << endl;

    // 4. Эмпирические данные для смесей
    cout << "4. Генерация эмпирических данных для смесей..." << endl;
    ofstream file_mixture_empirical("data_mixture_empirical.csv");
    file_mixture_empirical << "trivial,shift,scale,shape\n";

    for (int i = 0; i < mix_size; i++) {
        double sample_trivial = generate_mixture(&trivial);
        double sample_shift = generate_mixture(&shift);
        double sample_scale = generate_mixture(&scale_mix);
        double sample_shape = generate_mixture(&shape);

        file_mixture_empirical << sample_trivial << "," << sample_shift << "," << sample_scale << "," << sample_shape << "\n";
    }
    file_mixture_empirical.close();
    cout << "   Файл data_mixture_empirical.csv создан (" << mix_size << " элементов)" << endl;

    // 5. Эмпирическое распределение (3.3.1 - 3.3.2)
    cout << "5. Генерация данных для эмпирического распределения..." << endl;
    ofstream file_empirical_comparison("data_empirical_comparison.csv");
    file_empirical_comparison << "x,theoretical,empirical_100,empirical_1000,empirical_10000\n";

    const int size_100 = empirical_sizes[0];
    const int size_1000 = empirical_sizes[1];
    const int size_10000 = empirical_sizes[2];

    double* sample_100 = new double[size_100];
    double* sample_1000 = new double[size_1000];
    double* sample_10000 = new double[size_10000];

    // 6. Данные для тестирования класса Primary
    cout << "6. Генерация данных для класса Primary..." << endl;
    ofstream file_primary("data_primary_class.csv");
    file_primary << "x,pdf_class,pdf_function,moment,class_value,function_value,class_samples,function_samples\n";



    // Генерация данных для плотностей
    for (double x = -4.0; x <= 4.0; x += 0.04) {
        Primary prim_class(1.0, 2.0, 1.0); //тут менять в, лямбду, мю изначально было 0 1 1
        double pdf_class = prim_class.pdf(x);
        double pdf_function = pdf_main(x, 1.0, 2.0, 1.0); //тут менять мю, лямбду, в

        file_primary << x << "," << pdf_class << "," << pdf_function << ",,,,,\n";
    }

    // Данные для моментов
    Primary prim_moments(1.0, 2.0, 1.0); //тут менять в, лямбду, мю
    double class_mean, class_var, class_skew, class_kurt;
    double func_mean, func_var, func_skew, func_kurt;

    prim_moments.moments(&class_mean, &class_var, &class_skew, &class_kurt);
    moments_main(1.0, 2.0, 1.0, &func_mean, &func_var, &func_skew, &func_kurt);

    file_primary << ",,,Среднее," << class_mean << "," << func_mean << ",,\n";
    file_primary << ",,,Дисперсия," << class_var << "," << func_var << ",,\n";
    file_primary << ",,,Асимметрия," << class_skew << "," << func_skew << ",,\n";
    file_primary << ",,,Эксцесс," << class_kurt << "," << func_kurt << ",,\n";

    // Генерация случайных величин
    Primary prim_gen(1.0, 2.0, 1.0); //тут менять в, лямбду, мю 
    for (int i = 0; i < 1000; i++) {
        double class_sample = prim_gen.randNum();
        double func_sample = generate_main(1.0, 2.0, 1.0);
        file_primary << ",,,,,," << class_sample << "," << func_sample << "\n";
    }

    file_primary.close();
    cout << "   Файл data_primary_class.csv создан" << endl;

    // Заполнение выборок
    cout << "   Генерация выборок объемом " << size_100 << ", " << size_1000 << ", " << size_10000 << " элементов..." << endl;
    for (int i = 0; i < size_100; i++)
        sample_100[i] = generate_main(1.0, 2.0, 1.0);
    for (int i = 0; i < size_1000; i++)
        sample_1000[i] = generate_main(1.0, 2.0, 1.0);
    for (int i = 0; i < size_10000; i++)
        sample_10000[i] = generate_main(1.0, 2.0, 1.0);

    // Вычисление эмпирических PDF с помощью гистограмм
    cout << "   Вычисление эмпирических плотностей..." << endl;
    for (double x = -4.0; x <= 4.0; x += 0.08) {
        double theoretical = pdf_main(x, 0.0, 1.0, 1.0);

        double count_100 = 0, count_1000 = 0, count_10000 = 0;
        double bin_width = 0.3;

        for (int i = 0; i < size_100; i++) {
            if (fabs(sample_100[i] - x) < bin_width / 2) count_100++;
        }
        for (int i = 0; i < size_1000; i++) {
            if (fabs(sample_1000[i] - x) < bin_width / 2) count_1000++;
        }
        for (int i = 0; i < size_10000; i++) {
            if (fabs(sample_10000[i] - x) < bin_width / 2) count_10000++;
        }

        double empirical_100 = count_100 / (size_100 * bin_width);
        double empirical_1000 = count_1000 / (size_1000 * bin_width);
        double empirical_10000 = count_10000 / (size_10000 * bin_width);

        file_empirical_comparison << x << "," << theoretical << ","
            << empirical_100 << "," << empirical_1000 << "," << empirical_10000 << "\n";
    }

    file_empirical_comparison.close();
    cout << "   Файл data_empirical_comparison.csv создан" << endl;

    // Очистка
    delete[] sample_100;
    delete[] sample_1000;
    delete[] sample_10000;

    cout << "\nВсе файлы данных успешно созданы!" << endl;
    cout << "Файлы готовы для построения графиков в Python." << endl;
}

// Перегрузка для стандартных параметров
void generate_all_plot_data() {
    generate_all_plot_data(DEFAULT_SEN_SAMPLE_SIZE, DEFAULT_MIXTURE_SAMPLE_SIZE, const_cast<int*>(EMPIRICAL_SIZES));
}
/*
// Т Е С Т    З А Р А Б О Т А Л
// ТУТ ТОЖЕ МЕНЯТЬ ЛЯМБДЫ
void test_primary_class_menu() {
    cout << "\n Тестирование класса Primary" << endl;

    try {
        Primary test(0.0, 1.0, 1.0); //Тут
        cout << "Создан объект Primary с параметрами:" << endl;
        cout << "  v=" << test.getForm() << ", λ=" << test.getScale()
            << ", μ=" << test.getShift() << endl;

        double pdf_val = test.pdf(0.0);
        cout << "  f(0) = " << pdf_val << " (ожидалось ~0.55)" << endl;

        double mean, variance, skewness, kurtosis;
        test.moments(&mean, &variance, &skewness, &kurtosis);
        cout << "  Mξ = " << mean << " (ожидалось 0.0)" << endl;
        cout << "  Dξ = " << variance << " (ожидалось 0.596)" << endl;
        cout << "  γ₁ = " << skewness << " (ожидалось 0.0)" << endl;
        cout << "  γ₂ = " << kurtosis << " (ожидалось 0.405)" << endl;

        // Генерация нескольких значений
        cout << "  Пример генерации: ";
        for (int i = 0; i < 5; i++) {
            cout << test.randNum() << " ";
        }
        cout << endl;

        // Тест сохранения/загрузки
        cout << "  Тест сохранения/загрузки..." << endl;
        ofstream outFile("primary_test.txt");
        if (outFile) {
            test.save(outFile);
            outFile.close();
            cout << "  Объект сохранен в файл" << endl;
        }

        ifstream inFile("primary_test.txt");
        if (inFile) {
            Primary loaded(inFile);
            inFile.close();
            cout << "  Объект загружен из файла: v=" << loaded.getForm()
                << ", λ=" << loaded.getScale() << ", μ=" << loaded.getShift() << endl;
        }

        cout << "✓ Тест класса Primary выполнен успешно!" << endl;
    }
    catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
    }
}*/

// БАЗОВЫЕ ТЕСТЫ
void run_basic_tests() {
    cout << "\n Выполнение базовых тестов" << endl;

    // Test SEN(0,1,1) ТУТ МЕНЯТЬ ВХОДНЫЕ
    double mu = 5.0, lambda = 2.0, v = 1.0;
    double pdf_val = pdf_main(0.0, mu, lambda, v);
    double mean, variance, skewness, kurtosis;
    moments_main(mu, lambda, v, &mean, &variance, &skewness, &kurtosis);

    cout << "Тест SEN(0,1,1):" << endl;
    cout << "  f(0) = " << pdf_val << " (ожидалось 0.55)" << endl;
    cout << "  Математическое ожидание = " << mean << " (ожидалось 0.0)" << endl;
    cout << "  Дисперсия = " << variance << " (ожидалось 0.596)" << endl;
    cout << "  Эксцесс = " << kurtosis << " (ожидалось 0.405)" << endl;

    // Тест генерации
    cout << "  Тест генерации случайных величин..." << endl;
    double test_sum = 0.0;
    int test_size;
    cout << "  Введите количество генерируемых значений: ";
    cin >> test_size;
    if (test_size <= 0) test_size = 1000;

    for (int i = 0; i < test_size; i++) {
        test_sum += generate_main(0.0, 1.0, 1.0);
    }
    cout << "  Выборочное среднее = " << test_sum / test_size << endl;

    cout << "Базовые тесты завершены!" << endl;
}


// ПОЛНОЕ ТЕСТИРОВАНИЕ КЛАССА PRIMARY
void complete_primary_class_test() {
    cout << "\nПОЛНОЕ ТЕСТИРОВАНИЕ КЛАССА PRIMARY" << endl;

    try {
        // 1. Тестирование конструкторов и геттеров
        cout << "\n1. Тестирование конструкторов и геттеров" << endl;

        // 1.1. Конструктор с параметрами
        cout << "1.1. Конструктор с параметрами: Primary(1.0, 2.0, 3.0)" << endl;
        Primary dist_params(1.0, 2.0, 3.0);
        cout << "   - Состояние (v, lambda, mu): " << dist_params.getForm() << ", "
            << dist_params.getScale() << ", " << dist_params.getShift() << endl;

        // 1.2. Конструктор из потока
        cout << "1.2. Конструктор из потока" << endl;
        // Сохранение объекта в файл
        ofstream outFile("primary_test.txt");
        dist_params.save(outFile);
        outFile.close();

        // загрузка
        ifstream inFile("primary_test.txt");
        Primary dist_file(inFile);
        inFile.close();
        cout << "   - Состояние dist_file (v, lambda, mu): " << dist_file.getForm() << ", "
            << dist_file.getScale() << ", " << dist_file.getShift() << endl;

        // 2. Тестирование сеттеров
        cout << "\n2. Тестирование сеттеров" << endl;
        Primary dist_set(1.0, 1.0, 1.0);  // v=1.0, lambda=1.0, mu=1.0
        cout << "   - Начальное состояние:" << endl;
        cout << "     v (form) = " << dist_set.getForm() << endl;
        cout << "     lambda (scale) = " << dist_set.getScale() << endl;
        cout << "     mu (shift) = " << dist_set.getShift() << endl;

        // Изменяем параметры
        dist_set.setForm(5.0);    // v = 5.0
        dist_set.setScale(3.0);   // lambda = 3.0
        dist_set.setShift(2.0);   // mu = 2.0

        cout << "   - Конечное состояние:" << endl;
        cout << "     v (form) = " << dist_set.getForm() << endl;
        cout << "     lambda (scale) = " << dist_set.getScale() << endl;
        cout << "     mu (shift) = " << dist_set.getShift() << endl;

        // Проверяем корректность изменений
        if (dist_set.getForm() == 5.0 && dist_set.getScale() == 3.0 && dist_set.getShift() == 2.0) {
            cout << "Сеттеры работают корректно." << endl;
        }
        else {
            cout << "   >>> ОШИБКА: Сеттеры работают некорректно!" << endl;
        }

        // 3. Тестирование аналитических методов
        cout << "\n3. Тестирование аналитических методов" << endl;
        Primary dist_analysis(1.0, 2.0, 0.0); // Исправленные параметры
        cout << "   - Состояние dist_analysis (v, lambda, mu): " << dist_analysis.getForm() << ", "
            << dist_analysis.getScale() << ", " << dist_analysis.getShift() << endl;

        double mean, variance, skewness, kurtosis;
        dist_analysis.moments(&mean, &variance, &skewness, &kurtosis);
        cout << "   - Математическое ожидание (M): " << mean << endl;
        cout << "   - Дисперсия (D): " << variance << endl;
        cout << "   - Асимметрия (G1): " << skewness << endl;
        cout << "   - Эксцесс (G2): " << kurtosis << endl;

        double pdf_val = dist_analysis.pdf(0.0);
        cout << "   - Плотность в точке x = 0.0: " << pdf_val << endl;

        // 4. Тестирование генерации случайных чисел
        cout << "\n4. Тестирование генерации случайных чисел" << endl;
        cout << "   - Одиночное случайное число: " << dist_analysis.randNum() << endl;
        cout << "   - Вектор из 5 значений [Запуск 1]: ";
        for (int i = 0; i < 5; i++) {
            cout << dist_analysis.randNum() << " ";
        }
        cout << endl;

        cout << "Генерация работает." << endl;

        // 5. Тестирование обработки исключений
        cout << "\n5. Тестирование обработки исключений" << endl;

        // 5.1. Исключение для scale <= 0 в конструкторе
        try {
            Primary invalid_scale(1.0, -1.0, 0.0);
            cout << "   - ОШИБКА: Не сгенерировано исключение для scale <= 0!" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех: перехвачено исключение для scale <= 0: " << e.what() << endl;
        }

        // 5.2. Исключение для form <= 0 в конструкторе
        try {
            Primary invalid_form(-1.0, 1.0, 0.0);
            cout << "   - ОШИБКА: Не сгенерировано исключение для form <= 0" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех: перехвачено исключение для form <= 0: " << e.what() << endl;
        }

        // 5.3. Исключение в setScale
        try {
            Primary test_obj(1.0, 1.0, 0.0);
            test_obj.setScale(-1.0);
            cout << "   - ОШИБКА: Не сгенерировано исключение в setScale" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех (setScale): " << e.what() << endl;
        }

        // 5.4. Исключение в setForm
        try {
            Primary test_obj(1.0, 1.0, 0.0);
            test_obj.setForm(-1.0);
            cout << "   - ОШИБКА: Не сгенерировано исключение в setForm" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех (setForm): " << e.what() << endl;
        }

        // 5.5. Исключение при загрузке неверных параметров
        try {
            // Создаем файл с неверными параметрами
            ofstream badFile("bad_params.txt");
            badFile << "0.0 -1.0 1.0"; // lambda = -1.0
            badFile.close();

            ifstream badInFile("bad_params.txt");
            Primary bad_dist(badInFile);
            badInFile.close();
            cout << "   - ОШИБКА: Не сгенерировано исключение при загрузке" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех (загрузка): " << e.what() << endl;
        }

        // 6. Тестирование персистентности
        cout << "\n6. Тестирование персистентности" << endl;

        Primary original(2.0, 1.5, 1.0);

        // Сохраняем
        ofstream saveFile("persistence_test.txt");
        original.save(saveFile);
        saveFile.close();
        cout << "   - Объект сохранен: v=" << original.getForm()
            << ", lambda=" << original.getScale() << ", mu=" << original.getShift() << endl;

        // Загружаем
        ifstream loadFile("persistence_test.txt");
        Primary loaded(loadFile);
        loadFile.close();
        cout << "   - Объект загружен: v=" << loaded.getForm()
            << ", lambda=" << loaded.getScale() << ", mu=" << loaded.getShift() << endl;

        // Проверяем совпадение
        if (original.getForm() == loaded.getForm() &&
            original.getScale() == loaded.getScale() &&
            original.getShift() == loaded.getShift()) {
            cout << " Персистентность работает корректно" << endl;
        }
        else {
            cout << " Данные не совпадают после загрузки" << endl;
        }

        cout << "\n ПОЛНОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО" << endl;

    }
    catch (const exception& e) {
        cerr << "Ошибка при полном тестировании: " << e.what() << endl;
    }
}

// ТЕСТИРОВАНИЕ MIXTURE
void test_mixture_class() {
    cout << "\n Тестирование класса Mixture" << endl;

    try {
        // 1. Тестирование конструкторов
        cout << "1. Тестирование конструкторов:" << endl;

        Primary prim1(1.0, 1.0, 0.0);  // SEN(0,1,1)
        Primary prim2(1.0, 2.0, 3.0);  // SEN(3,2,1)

        // Конструктор с параметрами
        Mixture mixture(prim1, prim2, 0.7);
        cout << "   - Создана смесь с параметром p = " << mixture.getP() << endl;
        cout << "   - Компонент 1: v=" << mixture.component1().getForm()
            << ", lambda=" << mixture.component1().getScale()
            << ", mu=" << mixture.component1().getShift() << endl;
        cout << "   - Компонент 2: v=" << mixture.component2().getForm()
            << ", lambda=" << mixture.component2().getScale()
            << ", mu =" << mixture.component2().getShift() << endl;

        // Конструктор из файла
        ofstream outFile("mixture_test.txt");
        mixture.save(outFile);
        outFile.close();

        ifstream inFile("mixture_test.txt");
        Mixture loaded_mixture(inFile);
        inFile.close();
        cout << "   - Загружена смесь из файла: p = " << loaded_mixture.getP() << endl;

        // 2. Тестирование set-функций
        cout << "\n2. Тестирование set-функций:" << endl;
        mixture.setP(0.3);
        cout << "   - Новый параметр p = " << mixture.getP() << " (был 0.7)" << endl;

        // 3. Тестирование плотности
        cout << "\n3. Тестирование плотности:" << endl;
        double pdf_val = mixture.pdf(2.0);
        cout << "   - Плотность в точке x=2.0: " << pdf_val << endl;

        // Проверка в нескольких точках
        double test_points[] = { -2.0, 0.0, 2.0, 4.0 };
        for (double x : test_points) {
            cout << "     f(" << x << ") = " << mixture.pdf(x) << endl;
        }

        // 4. Тестирование моментов
        cout << "\n4. Тестирование моментов:" << endl;
        double mean, variance, skewness, kurtosis;
        mixture.moments(&mean, &variance, &skewness, &kurtosis);
        cout << "   - Математическое ожидание (Mξ) = " << mean << endl;
        cout << "   - Дисперсия (Dξ) = " << variance << endl;
        cout << "   - Асимметрия (γ₁) = " << skewness << endl;
        cout << "   - Эксцесс (γ₂) = " << kurtosis << endl;

        // 5. Тестирование генерации
        cout << "\n5. Тестирование генерации:" << endl;
        cout << "   - Одиночное значение: " << mixture.randNum() << endl;
        cout << "   - Вектор из 5 значений: ";
        for (int i = 0; i < 5; i++) {
            cout << mixture.randNum() << " ";
        }
        cout << endl;

        // 6. Тестирование исключений
        cout << "\n6. Тестирование исключений:" << endl;
        try {
            Mixture invalid_mixture(prim1, prim2, 1.5); // p > 1
            cout << "   - ОШИБКА: Не сгенерировано исключение для p > 1" << endl;
        }
        catch (const exception& e) {
            cout << "   - Успех: перехвачено исключение для p > 1: " << e.what() << endl;
        }

        try {
            mixture.setP(-0.5); // p < 0
            cout << "   - ОШИБКА: Не сгенерировано исключение в setP" << endl;
        }
        catch (const exception& e) {
            cout << "   - Успех (setP): " << e.what() << endl;
        }

        cout << "\n✓ Тест класса Mixture выполнен успешно!" << endl;

    }
    catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
    }
}

// ТЕСТИРОВАНИЕ КЛАССА EMPIRIC
void test_empiric_class() {
    cout << "\n Тестирование класса Empiric " << endl;

    try {
        // 1. Тестирование конструкторов
        cout << "1. Тестирование конструкторов:" << endl;

        Primary source_prim(1.0, 1.0, 0.0); // SEN(0,1,1)
        Primary prim2(1.0, 2.0, 1.0);
        Mixture source_mixture(source_prim, prim2, 0.5);

        // Конструктор из Primary
        Empiric empiric_from_prim(1000, source_prim, 20);
        cout << "   - Создано из Primary: n=" << empiric_from_prim.getN()
            << ", k=" << empiric_from_prim.getK() << endl;

        // Конструктор из Mixture
        Empiric empiric_from_mixture(800, source_mixture, 15);
        cout << "   - Создано из Mixture: n=" << empiric_from_mixture.getN()
            << ", k=" << empiric_from_mixture.getK() << endl;

        // Конструктор из Empiric
        Empiric empiric_from_empiric(500, empiric_from_prim, 10);
        cout << "   - Создано из Empiric: n=" << empiric_from_empiric.getN()
            << ", k=" << empiric_from_empiric.getK() << endl;

        // 2. Тестирование диапазонов данных
        cout << "\n2. Тестирование диапазонов данных:" << endl;
        cout << "   - Empiric из Primary: [" << empiric_from_prim.getMin()
            << ", " << empiric_from_prim.getMax() << "]" << endl;
        cout << "   - Empiric из Mixture: [" << empiric_from_mixture.getMin()
            << ", " << empiric_from_mixture.getMax() << "]" << endl;

        // 3. Тестирование плотности
        cout << "\n3. Тестирование плотности:" << endl;
        double pdf_prim = empiric_from_prim.pdf(0.0);
        double pdf_mixture = empiric_from_mixture.pdf(1.0);
        cout << "   - Плотность Empiric(Primary) в x=0.0: " << pdf_prim << endl;
        cout << "   - Плотность Empiric(Mixture) в x=1.0: " << pdf_mixture << endl;

        // 4. Тестирование моментов
        cout << "\n4. Тестирование моментов:" << endl;
        double mean_prim, var_prim, skew_prim, kurt_prim;
        double mean_mix, var_mix, skew_mix, kurt_mix;

        empiric_from_prim.moments(&mean_prim, &var_prim, &skew_prim, &kurt_prim);
        empiric_from_mixture.moments(&mean_mix, &var_mix, &skew_mix, &kurt_mix);

        cout << "   - Empiric(Primary): M=" << mean_prim << ", D=" << var_prim
            << ", сим=" << skew_prim << ", экс=" << kurt_prim << endl;
        cout << "   - Empiric(Mixture): M=" << mean_mix << ", D=" << var_mix
            << ", сим=" << skew_mix << ", экс=" << kurt_mix << endl;

        // 5. Тестирование генерации
        cout << "\n5. Тестирование генерации:" << endl;
        cout << "   - Empiric(Primary): ";
        for (int i = 0; i < 3; i++) {
            cout << empiric_from_prim.randNum() << " ";
        }
        cout << endl;
        cout << "   - Empiric(Mixture): ";
        for (int i = 0; i < 3; i++) {
            cout << empiric_from_mixture.randNum() << " ";
        }
        cout << endl;

        // 6. Тестирование глубокого копирования
        cout << "\n6. Тестирование глубокого копирования:" << endl;

        // Конструктор копирования
        Empiric copy_constructed(empiric_from_prim);
        cout << "   - Конструктор копирования: n=" << copy_constructed.getN()
            << ", k=" << copy_constructed.getK() << endl;

        // Оператор присваивания
        Empiric assigned;
        assigned = empiric_from_prim;
        cout << "   - Оператор присваивания: n=" << assigned.getN()
            << ", k=" << assigned.getK() << endl;

        // Проверка независимости копий
        double original_pdf = empiric_from_prim.pdf(0.0);
        double copy_pdf = copy_constructed.pdf(0.0);
        cout << "   - Проверка независимости: original.pdf(0)=" << original_pdf
            << ", copy.pdf(0)=" << copy_pdf << endl;

        if (abs(original_pdf - copy_pdf) < 1e-10) {
            cout << "Копирование работает корректно!" << endl;
        }
        else {
            cout << "Данные не совпадают после копирования!" << endl;
        }

        // 7. Тестирование исключений
        cout << "\n7. Тестирование исключений:" << endl;
        try {
            Empiric invalid_empiric(1, source_prim, 10); // n=1
            cout << "   - ОШИБКА: Не сгенерировано исключение для n=1!" << endl;
        }
        catch (const exception& e) {
            cout << "   - Успех: перехвачено исключение для n=1: " << e.what() << endl;
        }

        cout << "\nТест класса Empiric выполнен успешно" << endl;

    }
    catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
    }
}

//  ДЕМОНСТРАЦИЯ ОТНОШЕНИЙ МЕЖДУ ОБЪЕКТАМИ

void demonstrate_objects_relationships() {
    cout << "\n Демонстрация отношений между объектами" << endl;

    try {
        // 1. Композиция: Mixture содержит Primary объекты
        cout << "1. Композиция (Mixture содержит Primary):" << endl;

        Primary comp1(1.0, 1.0, 0.0);  // SEN(0,1,1)
        Primary comp2(1.0, 2.0, 1.0);  // SEN(1,2,1)
        Mixture mixture(comp1, comp2, 0.6);

        cout << "   Создана смесь с двумя компонентами Primary" << endl;
        cout << "   Время жизни компонентов совпадает со временем жизни смеси" << endl;
        cout << "   Компонент 1: mu=" << mixture.component1().getShift()
            << ", lambda=" << mixture.component1().getScale() << endl;
        cout << "   - Компонент 2: mu=" << mixture.component2().getShift()
            << ", lambda=" << mixture.component2().getScale() << endl;

        // 2. Клиент-серверные отношения: Empiric использует другие распределения
        cout << "\n2. Клиент-серверные отношения (Empiric использует распределения):" << endl;

        // Empiric из Primary
        Empiric empiric_prim(500, comp1, 15);
        cout << "   - Empiric использует Primary для генерации выборки" << endl;
        cout << "     Объем выборки: " << empiric_prim.getN() << endl;

        // Empiric из Mixture  
        Empiric empiric_mixture(500, mixture, 15);
        cout << "   - Empiric использует Mixture для генерации выборки" << endl;
        cout << "     Объем выборки: " << empiric_mixture.getN() << endl;

        // 3. Связи между Empiric объектами
        cout << "\n3. Связи между Empiric объектами:" << endl;

        Empiric empiric_from_empiric(200, empiric_prim, 10);
        cout << "   - Empiric использует другой Empiric для генерации выборки" << endl;
        cout << "     Цепочка: Primary > Empiric1 > Empiric2" << endl;
        cout << "     Размеры: " << empiric_prim.getN() << " > "
            << empiric_from_empiric.getN() << " элементов" << endl;

        // 4. Демонстрация глубокого копирования
        cout << "\n4. Глубокое копирование (Empiric):" << endl;

        Empiric original(100, comp1, 10);
        cout << "   - Оригинал: n=" << original.getN() << ", k=" << original.getK() << endl;

        // Конструктор копирования
        Empiric copy_constructed(original);
        cout << "   - Конструктор копирования: n=" << copy_constructed.getN()
            << ", k=" << copy_constructed.getK() << endl;

        // Оператор присваивания
        Empiric assigned;
        assigned = original;
        cout << "   - Оператор присваивания: n=" << assigned.getN()
            << ", k=" << assigned.getK() << endl;

        // Проверка, что это разные объекты
        cout << "   - Адреса объектов разные (глубокое копирование):" << endl;
        cout << "     Оригинал: " << &original << endl;
        cout << "     Копия:    " << &copy_constructed << endl;

        // 5. Демонстрация работы с разными типами распределений
        cout << "\n5. Работа с разными типами распределений:" << endl;

        cout << "   - Primary SEN(0,1,1):" << endl;
        cout << "     M=" << comp1.getShift() << ", lambda=" << comp1.getScale()
            << ", v=" << comp1.getForm() << endl;

        cout << "   - Mixture (p=0.6):" << endl;
        double mean, variance, skewness, kurtosis;
        mixture.moments(&mean, &variance, &skewness, &kurtosis);
        cout << "     м. ожид. =" << mean << ", дисперсия=" << variance << endl;

        cout << "   - Empiric из Primary:" << endl;
        empiric_prim.moments(&mean, &variance, &skewness, &kurtosis);
        cout << "     м. ожид=" << mean << ", дисперсия=" << variance << endl;

    }
    catch (const exception& e) {
        cerr << "Ошибка при демонстрации отношений: " << e.what() << endl;
    }
}

// ГЛАВНАЯ ФУНКЦИЯ
int main() {
    setlocale(LC_ALL, "Russian");
    ;
    int choice;
    // В главной функции main() обновить меню:
    do {
        cout << "\nГЛАВНОЕ МЕНЮ" << endl;
        cout << "1 - Выполнить базовые тесты" << endl;
        cout << "2 - Сгенерировать данные для графиков (стандартные размеры)" << endl;
        cout << "3 - Сгенерировать данные для графиков (пользовательские размеры)" << endl;
        cout << "4 - Тестирование эмпирического распределения" << endl;
        cout << "5 - Тестирование класса Primary" << endl;
        cout << "6 - Тестирование класса Mixture" << endl;
        cout << "7 - Тестирование класса Empiric" << endl;
        cout << "8 - Тест отношений между объектами" << endl;
        cout << "0 - Выход" << endl;
        cout << "Выберите операцию: ";
        cin >> choice;

        switch (choice) {
        case 1:
            run_basic_tests();
            break;
        case 2:
            generate_all_plot_data();
            break;
        case 3:
        {
            int sen_size, mix_size;
            int empirical_sizes[3];
            input_sample_sizes(sen_size, mix_size, empirical_sizes);
            generate_all_plot_data(sen_size, mix_size, empirical_sizes);
        }
        break;
        case 4:
            test_empirical_distribution();
            break;
        case 5:
            complete_primary_class_test();
            break;
        case 6:
            test_mixture_class();
            break;
        case 7:
            test_empiric_class();
            break;
       case 8:
            demonstrate_objects_relationships();
            break;
        case 0:
            cout << "Выход из программы..." << endl;
            break;
        default:
            cout << "Неверный выбор! Попробуйте снова." << endl;
        }
    } while (choice != 0);

    cout << "\nПрограмма завершена." << endl;
    return 0;
}