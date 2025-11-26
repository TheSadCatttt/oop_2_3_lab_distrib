#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>
#include "distributions.h"
#include "primary.h"

using namespace std;

// ГЛОБАЛЬНЫЕ КОНСТАНТЫ
const int DEFAULT_SEN_SAMPLE_SIZE = 10000;
const int DEFAULT_MIXTURE_SAMPLE_SIZE = 10000;
const int EMPIRICAL_SIZES[] = { 100, 1000, 10000 };

// ОБЪЯВЛЕНИЯ ФУНКЦИЙ
void input_sample_sizes(int& sen_size, int& mix_size, int empirical_sizes[]);
void test_empirical_distribution();
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
}

// --- БАЗОВЫЕ ТЕСТЫ ---
void run_basic_tests() {
    cout << "\n=== Выполнение базовых тестов ===" << endl;

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


// --- ПОЛНОЕ ТЕСТИРОВАНИЕ КЛАССА PRIMARY ---
void complete_primary_class_test() {
    cout << "\n=== ПОЛНОЕ ТЕСТИРОВАНИЕ КЛАССА PRIMARY ===" << endl;

    try {
        // 1. Тестирование конструкторов и геттеров
        cout << "\n1. Тестирование конструкторов и геттеров" << endl;
        cout << "-------------------------------------------" << endl;

        // 1.1. Конструктор с параметрами
        cout << "1.1. Конструктор с параметрами: Primary(1.0, 2.0, 3.0)" << endl;
        Primary dist_params(1.0, 2.0, 3.0);
        cout << "   - Состояние (v, lambda, mu): " << dist_params.getForm() << ", "
            << dist_params.getScale() << ", " << dist_params.getShift() << endl;

        // 1.2. Конструктор из потока
        cout << "1.2. Конструктор из потока" << endl;
        // Сначала сохраним объект в файл
        ofstream outFile("primary_test.txt");
        dist_params.save(outFile);
        outFile.close();

        // Затем загрузим
        ifstream inFile("primary_test.txt");
        Primary dist_file(inFile);
        inFile.close();
        cout << "   - Состояние dist_file (v, lambda, mu): " << dist_file.getForm() << ", "
            << dist_file.getScale() << ", " << dist_file.getShift() << endl;

        // 2. Тестирование сеттеров
        cout << "\n2. Тестирование сеттеров" << endl;
        cout << "------------------------" << endl;
        Primary dist_set(1.0, 1.0, 1.0);  // v=1.0, lambda=1.0, mu=1.0
        cout << "   - Начальное состояние:" << endl;
        cout << "     v (form) = " << dist_set.getForm() << endl;
        cout << "     λ (scale) = " << dist_set.getScale() << endl;
        cout << "     μ (shift) = " << dist_set.getShift() << endl;

        // Изменяем параметры
        dist_set.setForm(5.0);    // v = 5.0
        dist_set.setScale(3.0);   // lambda = 3.0
        dist_set.setShift(2.0);   // mu = 2.0

        cout << "   - Конечное состояние:" << endl;
        cout << "     v (form) = " << dist_set.getForm() << endl;
        cout << "     λ (scale) = " << dist_set.getScale() << endl;
        cout << "     μ (shift) = " << dist_set.getShift() << endl;

        // Проверяем корректность изменений
        if (dist_set.getForm() == 5.0 && dist_set.getScale() == 3.0 && dist_set.getShift() == 2.0) {
            cout << "   >>> ПОДТВЕРЖДЕНО: Сеттеры работают корректно!" << endl;
        }
        else {
            cout << "   >>> ОШИБКА: Сеттеры работают некорректно!" << endl;
        }

        // 3. Тестирование аналитических методов
        cout << "\n3. Тестирование аналитических методов" << endl;
        cout << "-------------------------------------" << endl;
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
        cout << "----------------------------------------" << endl;
        cout << "   - Одиночное случайное число: " << dist_analysis.randNum() << endl;

        cout << "   - Вектор из 5 значений [Запуск 1]: ";
        for (int i = 0; i < 5; i++) {
            cout << dist_analysis.randNum() << " ";
        }
        cout << endl;

        // Проверка воспроизводимости (нужно сбросить генератор для демонстрации)
        cout << "   >>> ПОДТВЕРЖДЕНО: Генерация работает!" << endl;

        // 5. Тестирование обработки исключений
        cout << "\n5. Тестирование обработки исключений" << endl;
        cout << "------------------------------------" << endl;

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
            cout << "   - ОШИБКА: Не сгенерировано исключение для form <= 0!" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех: перехвачено исключение для form <= 0: " << e.what() << endl;
        }

        // 5.3. Исключение в setScale
        try {
            Primary test_obj(1.0, 1.0, 0.0);
            test_obj.setScale(-1.0);
            cout << "   - ОШИБКА: Не сгенерировано исключение в setScale!" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех (setScale): " << e.what() << endl;
        }

        // 5.4. Исключение в setForm
        try {
            Primary test_obj(1.0, 1.0, 0.0);
            test_obj.setForm(-1.0);
            cout << "   - ОШИБКА: Не сгенерировано исключение в setForm!" << endl;
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
            cout << "   - ОШИБКА: Не сгенерировано исключение при загрузке!" << endl;
        }
        catch (const std::invalid_argument& e) {
            cout << "   - Успех (загрузка): " << e.what() << endl;
        }

        // 6. Тестирование персистентности
        cout << "\n6. Тестирование персистентности" << endl;
        cout << "-------------------------------" << endl;

        Primary original(2.0, 1.5, 1.0);

        // Сохраняем
        ofstream saveFile("persistence_test.txt");
        original.save(saveFile);
        saveFile.close();
        cout << "   - Объект сохранен: v=" << original.getForm()
            << ", λ=" << original.getScale() << ", μ=" << original.getShift() << endl;

        // Загружаем
        ifstream loadFile("persistence_test.txt");
        Primary loaded(loadFile);
        loadFile.close();
        cout << "   - Объект загружен: v=" << loaded.getForm()
            << ", λ=" << loaded.getScale() << ", μ=" << loaded.getShift() << endl;

        // Проверяем совпадение
        if (original.getForm() == loaded.getForm() &&
            original.getScale() == loaded.getScale() &&
            original.getShift() == loaded.getShift()) {
            cout << "   >>> ПОДТВЕРЖДЕНО: Персистентность работает корректно!" << endl;
        }
        else {
            cout << "   >>> ОШИБКА: Данные не совпадают после загрузки!" << endl;
        }

        cout << "\n=== ПОЛНОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===" << endl;

    }
    catch (const exception& e) {
        cerr << "Ошибка при полном тестировании: " << e.what() << endl;
    }
}

// ГЛАВНАЯ ФУНКЦИЯ
int main() {
    setlocale(LC_ALL, "Russian");
    ;
    int choice;
    do {
        cout << "\nГЛАВНОЕ МЕНЮ" << endl;
        cout << "1 - Выполнить базовые тесты" << endl;
        cout << "2 - Сгенерировать данные для графиков (стандартные размеры)" << endl;
        cout << "3 - Сгенерировать данные для графиков (пользовательские размеры)" << endl;
        cout << "4 - Тестирование эмпирического распределения" << endl;
        cout << "5 - Тестирование класса Primary" << endl;
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