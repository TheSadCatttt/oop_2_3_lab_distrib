#include "Empiric.h"
#include <stdexcept>

// Конструктор из Primary
Empiric::Empiric(int n0, const Primary& prim, int k0)
    : n(n0 > 1 ? n0 : throw std::invalid_argument("Sample size must be greater than 1")),
    data(new double[n]), fr(nullptr) {

    // Генерация выборки
    for (int i = 0; i < n; i++) {
        data[i] = prim.randNum();
    }

    init_empirical(data, n, k0);
}

// Конструктор из Mixture
Empiric::Empiric(int n0, const Mixture& mixt, int k0)
    : n(n0 > 1 ? n0 : throw std::invalid_argument("Sample size must be greater than 1")),
    data(new double[n]), fr(nullptr) {

    // Генерация выборки
    for (int i = 0; i < n; i++) {
        data[i] = mixt.randNum();
    }

    init_empirical(data, n, k0);
}

// Конструктор из Empiric
Empiric::Empiric(int n0, const Empiric& emp, int k0)
    : n(n0 > 1 ? n0 : throw std::invalid_argument("Sample size must be greater than 1")),
    data(new double[n]), fr(nullptr) {

    // Генерация выборки
    for (int i = 0; i < n; i++) {
        data[i] = emp.randNum();
    }

    init_empirical(data, n, k0);
}

// Конструктор копирования
Empiric::Empiric(const Empiric& other)
    : n(other.n), k(other.k), min_val(other.min_val), max_val(other.max_val),
    data(new double[other.n]), fr(new double[other.k]) {

    std::copy(other.data, other.data + other.n, data);
    std::copy(other.fr, other.fr + other.k, fr);
}
// Конструктор по умолчанию
Empiric::Empiric()
    : n(0), k(0), min_val(0.0), max_val(0.0), data(nullptr), fr(nullptr) {
}
// Оператор присваивания
Empiric& Empiric::operator=(const Empiric& other) {
    if (this != &other) {
        free_memory();
        copy_data(other);
    }
    return *this;
}

// Деструктор
Empiric::~Empiric() {
    free_memory();
}

// Инициализация эмпирического распределения
void Empiric::init_empirical(double* sample_data, int sample_size, int k0) {
    // Сортировка данных
    std::sort(data, data + n);

    min_val = data[0];
    max_val = data[n - 1];

    // Формула Стёрджеса для выбора числа интервалов
    if (k0 <= 0) {
        k = static_cast<int>(std::floor(std::log2(n))) + 1;
    }
    else {
        k = k0;
    }

    // Инициализация частот
    fr = new double[k];
    double bin_width = (max_val - min_val) / k;

    // Подсчет частот
    std::fill(fr, fr + k, 0.0);
    for (int i = 0; i < n; i++) {
        int bin_index = static_cast<int>((data[i] - min_val) / bin_width);
        if (bin_index == k) bin_index = k - 1;
        fr[bin_index] += 1.0;
    }

    // Нормализация частот
    for (int i = 0; i < k; i++) {
        fr[i] /= (n * bin_width);
    }
}

// Плотность эмпирического распределения
double Empiric::pdf(double x) const {
    if (x < min_val || x > max_val) return 0.0;

    double bin_width = (max_val - min_val) / k;
    int bin_index = static_cast<int>((x - min_val) / bin_width);

    if (bin_index == k) bin_index = k - 1;
    return fr[bin_index];
}

// Моменты эмпирического распределения
void Empiric::moments(double* mean, double* variance, double* skewness, double* kurtosis) const {
    // Выборочное среднее
    *mean = std::accumulate(data, data + n, 0.0) / n;

    // Выборочная дисперсия и высшие моменты
    double sum_sq = 0.0, sum_cube = 0.0, sum_quad = 0.0;
    for (int i = 0; i < n; i++) {
        double dev = data[i] - *mean;
        double dev_sq = dev * dev;
        sum_sq += dev_sq;
        sum_cube += dev_sq * dev;
        sum_quad += dev_sq * dev_sq;
    }

    *variance = sum_sq / (n - 1);

    // Коэффициент асимметрии
    if (*variance > 0) {
        *skewness = (sum_cube / n) / std::pow(*variance, 1.5);
    }
    else {
        *skewness = 0.0;
    }

    // Коэффициент эксцесса
    if (*variance > 0) {
        *kurtosis = (sum_quad / n) / (*variance * *variance) - 3.0;
    }
    else {
        *kurtosis = 0.0;
    }
}

// Генерация случайной величины
double Empiric::randNum() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> unif(0.0, 1.0);

    double bin_width = (max_val - min_val) / k;

    // Выбор бина согласно относительным частотам
    double r = unif(gen);
    double cumulative = 0.0;
    int selected_bin = 0;

    for (int i = 0; i < k; i++) {
        double prob = fr[i] * bin_width;
        cumulative += prob;
        if (r <= cumulative) {
            selected_bin = i;
            break;
        }
    }

    // Генерация равномерно в выбранном бине
    double left = min_val + selected_bin * bin_width;
    double right = left + bin_width;

    return left + unif(gen) * bin_width;
}

// Сохранение в поток
void Empiric::save(std::ostream& out) const {
    out << n << " " << k << " " << min_val << " " << max_val << " ";
    for (int i = 0; i < n; i++) {
        out << data[i] << " ";
    }
    for (int i = 0; i < k; i++) {
        out << fr[i] << " ";
    }
}

// Загрузка из потока
void Empiric::load(std::istream& in) {
    free_memory();

    in >> n >> k >> min_val >> max_val;

    data = new double[n];
    fr = new double[k];

    for (int i = 0; i < n; i++) {
        in >> data[i];
    }
    for (int i = 0; i < k; i++) {
        in >> fr[i];
    }
}

// Освобождение памяти
void Empiric::free_memory() {
    delete[] data;
    delete[] fr;
    data = nullptr;
    fr = nullptr;
    n = 0;
    k = 0;
}

// Копирование данных
void Empiric::copy_data(const Empiric& other) {
    n = other.n;
    k = other.k;
    min_val = other.min_val;
    max_val = other.max_val;

    data = new double[n];
    fr = new double[k];

    std::copy(other.data, other.data + n, data);
    std::copy(other.fr, other.fr + k, fr);
}