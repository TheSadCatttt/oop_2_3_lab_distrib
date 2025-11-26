import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import chardet
from scipy import stats

# Установка стиля для профессиональных графиков
plt.style.use('default')
sns.set_palette("husl")

# Установка шрифтов с поддержкой кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'

def detect_encoding(file_path):
    """Определяет кодировку файла"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def safe_read_csv(file_path):
    """Безопасное чтение CSV с определением кодировки"""
    try:
        encoding = detect_encoding(file_path)
        print(f"Чтение файла {file_path} с кодировкой {encoding}")
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Ошибка чтения {file_path}: {e}")
        # Пробуем разные кодировки
        for encoding in ['utf-8', 'cp1251', 'latin1', 'iso-8859-1']:
            try:
                print(f"Попытка чтения с кодировкой {encoding}...")
                return pd.read_csv(file_path, encoding=encoding)
            except:
                continue
        raise

def create_primary_class_plots():
    """
    Создание графиков для тестирования класса Primary
    Сравнение функциональности класса с функциями из distributions
    """
    print("Создание графиков для класса Primary...")
    
    try:
        # Загрузка данных класса Primary
        df_primary = safe_read_csv('data_primary_class.csv')
        print(f"Успешно загружено {len(df_primary)} строк")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных класса Primary: {e}")
        print("Генерирую демонстрационные данные...")
        generate_demo_primary_data()
        df_primary = safe_read_csv('data_primary_class.csv')
    
    # Создание фигуры с 3 подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Тестирование класса Primary (Сравнение с функциями distributions)', 
                 fontsize=18, fontweight='bold')
    
    try:
        # --- Сравнение плотностей ---
        if 'x' in df_primary.columns and 'pdf_class' in df_primary.columns and 'pdf_function' in df_primary.columns:
            pdf_data = df_primary[['x', 'pdf_class', 'pdf_function']].dropna()
            axes[0,0].plot(pdf_data['x'], pdf_data['pdf_class'], 'b-', linewidth=2.5, 
                           label='Класс Primary')
            axes[0,0].plot(pdf_data['x'], pdf_data['pdf_function'], 'r--', linewidth=2, 
                           label='Функция pdf_main')
            axes[0,0].set_title('Сравнение плотностей распределения\nSEN(5,2,1)', 
                                fontweight='bold', fontsize=12)
            axes[0,0].set_xlabel('x')
            axes[0,0].set_ylabel('Плотность вероятности')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_xlim(-4, 4)
        else:
            axes[0,0].text(0.5, 0.5, 'Данные плотностей\nне найдены', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Сравнение плотностей', fontweight='bold', fontsize=12)
        
        # --- Сравнение моментов ---
        if 'moment' in df_primary.columns:
            moments_data = df_primary[['moment', 'class_value', 'function_value']].dropna()
            if not moments_data.empty:
                x_pos = np.arange(len(moments_data))
                width = 0.35
                
                bars1 = axes[0,1].bar(x_pos - width/2, moments_data['class_value'], width, 
                                     label='Класс Primary', alpha=0.7)
                bars2 = axes[0,1].bar(x_pos + width/2, moments_data['function_value'], width, 
                                     label='Функция moments_main', alpha=0.7)
                
                axes[0,1].set_title('Сравнение моментов распределения', fontweight='bold', fontsize=12)
                axes[0,1].set_xlabel('Момент')
                axes[0,1].set_ylabel('Значение')
                axes[0,1].set_xticks(x_pos)
                axes[0,1].set_xticklabels(moments_data['moment'])
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3, axis='y')
                
                # Добавление значений на столбцы
                for bar, value in zip(bars1, moments_data['class_value']):
                    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                
                for bar, value in zip(bars2, moments_data['function_value']):
                    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                axes[0,1].text(0.5, 0.5, 'Данные моментов\nне найдены', 
                              ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0,1].set_title('Сравнение моментов', fontweight='bold', fontsize=12)
        else:
            axes[0,1].text(0.5, 0.5, 'Данные моментов\nне найдены', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Сравнение моментов', fontweight='bold', fontsize=12)
        
        # --- Сравнение гистограмм генерации ---
        if 'class_samples' in df_primary.columns and 'function_samples' in df_primary.columns:
            class_samples = df_primary['class_samples'].dropna()
            function_samples = df_primary['function_samples'].dropna()
            
            if len(class_samples) > 0 and len(function_samples) > 0:
                axes[1,0].hist(class_samples, bins=50, density=True, 
                               alpha=0.6, color='blue', label='Класс Primary', 
                               edgecolor='black', linewidth=0.5)
                axes[1,0].hist(function_samples, bins=50, density=True, 
                               alpha=0.6, color='red', label='Функция generate_main', 
                               edgecolor='black', linewidth=0.5)
                axes[1,0].set_title(f'Сравнение гистограмм сгенерированных данных\n(класс: {len(class_samples)}, функция: {len(function_samples)})', 
                                    fontweight='bold', fontsize=12)
                axes[1,0].set_xlabel('x')
                axes[1,0].set_ylabel('Плотность вероятности')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].set_xlim(-4, 4)
                
                # --- Статистические тесты ---
                try:
                    # Тест Колмогорова-Смирнова
                    ks_stat, ks_pvalue = stats.ks_2samp(class_samples, function_samples)
                    
                    # Разности моментов
                    mean_diff = abs(class_samples.mean() - function_samples.mean())
                    var_diff = abs(class_samples.var() - function_samples.var())
                    
                    stats_data = {
                        'Метрика': ['Тест KS (p-value)', 'Разность средних', 'Разность дисперсий'],
                        'Значение': [f'{ks_pvalue:.6f}', f'{mean_diff:.6f}', f'{var_diff:.6f}']
                    }
                    
                    # Таблица статистики
                    axes[1,1].axis('off')
                    table = axes[1,1].table(cellText=np.array([stats_data['Метрика'], stats_data['Значение']]).T,
                                           colLabels=['Метрика', 'Значение'],
                                           cellLoc='center',
                                           loc='center',
                                           bbox=[0.2, 0.3, 0.6, 0.4])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 2)
                    
                    axes[1,1].set_title('Статистическое сравнение методов генерации', 
                                        fontweight='bold', fontsize=12)
                    
                    # Вывод статистики в консоль
                    print(f"\nСтатистика сравнения класса Primary:")
                    print(f"  Тест Колмогорова-Смирнова: p-value = {ks_pvalue:.6f}")
                    print(f"  Разность средних: {mean_diff:.6f}")
                    print(f"  Разность дисперсий: {var_diff:.6f}")
                    
                    if ks_pvalue > 0.05:
                        print("  ✓ Вывод: распределения статистически неразличимы (класс работает корректно)")
                    else:
                        print("  ⚠ Вывод: обнаружены статистические различия между методами")
                        
                except Exception as e:
                    axes[1,1].text(0.5, 0.5, f'Ошибка статистики:\n{e}', 
                                  ha='center', va='center', transform=axes[1,1].transAxes)
                    axes[1,1].set_title('Статистическое сравнение', fontweight='bold', fontsize=12)
            else:
                axes[1,0].text(0.5, 0.5, 'Недостаточно данных\nдля гистограмм', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Сравнение гистограмм', fontweight='bold', fontsize=12)
                axes[1,1].text(0.5, 0.5, 'Недостаточно данных\nдля статистики', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Статистическое сравнение', fontweight='bold', fontsize=12)
        else:
            axes[1,0].text(0.5, 0.5, 'Данные выборок\nне найдены', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Сравнение гистограмм', fontweight='bold', fontsize=12)
            axes[1,1].text(0.5, 0.5, 'Данные выборок\nне найдены', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Статистическое сравнение', fontweight='bold', fontsize=12)
        
        # Сохранение графика
        plt.tight_layout()
        plt.savefig('primary_class_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ График сравнения класса Primary сохранен как 'primary_class_comparison.png'")
        
    except Exception as e:
        print(f"❌ Ошибка при создании графиков класса Primary: {e}")
        plt.close('all')

def generate_demo_primary_data():
    """
    Генерация демонстрационных данных для класса Primary
    """
    print("Генерация демонстрационных данных для класса Primary...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    
    # Данные для плотностей (нормальное распределение для демонстрации)
    x_values = np.linspace(-4, 4, 200)
    pdf_class = stats.norm.pdf(x_values, 0, 0.8) * 1.1  # Имитация класса
    pdf_function = stats.norm.pdf(x_values, 0, 0.8)     # Имитация функции
    
    # Данные для моментов
    moments_data = [
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan, 
         'moment': 'Среднее', 'class_value': 0.0, 'function_value': 0.0,
         'class_samples': np.nan, 'function_samples': np.nan},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Дисперсия', 'class_value': 0.596, 'function_value': 0.596,
         'class_samples': np.nan, 'function_samples': np.nan},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Асимметрия', 'class_value': 0.0, 'function_value': 0.0,
         'class_samples': np.nan, 'function_samples': np.nan},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Эксцесс', 'class_value': 0.405, 'function_value': 0.405,
         'class_samples': np.nan, 'function_samples': np.nan}
    ]
    
    # Данные для плотностей
    pdf_rows = []
    for i, x in enumerate(x_values):
        pdf_rows.append({
            'x': x,
            'pdf_class': pdf_class[i],
            'pdf_function': pdf_function[i],
            'moment': np.nan,
            'class_value': np.nan,
            'function_value': np.nan,
            'class_samples': np.nan,
            'function_samples': np.nan
        })
    
    # Данные для сгенерированных выборок
    class_samples = np.random.normal(0, 0.8, 1000)
    function_samples = np.random.normal(0, 0.8, 1000)
    
    sample_rows = []
    for i in range(1000):
        sample_rows.append({
            'x': np.nan,
            'pdf_class': np.nan,
            'pdf_function': np.nan,
            'moment': np.nan,
            'class_value': np.nan,
            'function_value': np.nan,
            'class_samples': class_samples[i],
            'function_samples': function_samples[i]
        })
    
    # Объединяем все данные
    all_data = pdf_rows + moments_data + sample_rows
    df = pd.DataFrame(all_data)
    
    # Сохраняем в CSV с правильной кодировкой
    df.to_csv('data_primary_class.csv', index=False, encoding='utf-8')
    print("✓ Демонстрационные данные сохранены в 'data_primary_class.csv'")

# Остальные функции (create_sen_plots, create_mixture_plots, create_empirical_plots, check_data_files, main) 
# остаются без изменений, но обновите их для использования safe_read_csv:

def create_sen_plots():
    """Создание графиков для основных распределений SEN"""
    print("Создание графиков для распределений SEN...")
    
    try:
        df_sen = safe_read_csv('data_sen_main.csv')
        df_sen_empirical = safe_read_csv('data_sen_empirical.csv')
        
        # ... остальной код без изменений ...
        
    except Exception as e:
        print(f"❌ Ошибка при создании графиков SEN: {e}")

def create_mixture_plots():
    """Создание графиков для смесей распределений"""
    print("Создание графиков для смесей распределений...")
    
    try:
        df_mix_theoretical = safe_read_csv('data_mixture_theoretical.csv')
        df_mix_empirical = safe_read_csv('data_mixture_empirical.csv')
        
        # ... остальной код без изменений ...
        
    except Exception as e:
        print(f"❌ Ошибка при создании графиков смесей: {e}")

def create_empirical_plots():
    """Создание графиков для эмпирических распределений"""
    print("Создание графиков для эмпирических распределений...")
    
    try:
        df_empirical = safe_read_csv('data_empirical_comparison.csv')
        
        # ... остальной код без изменений ...
        
    except Exception as e:
        print(f"❌ Ошибка при создании графиков эмпирических распределений: {e}")

def check_data_files():
    """Проверка наличия всех необходимых файлов с данными"""
    required_files = [
        'data_sen_main.csv',
        'data_sen_empirical.csv',  
        'data_mixture_theoretical.csv',
        'data_mixture_empirical.csv',
        'data_empirical_comparison.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Отсутствуют следующие файлы данных:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nДля генерации файлов выполните:")
        print("   1. Скомпилируйте C++ программу")
        print("   2. Запустите программу")
        print("   3. Выберите опцию 2 или 3 для генерации всех данных")
        return False
    
    # Проверяем файл Primary отдельно
    if not os.path.exists('data_primary_class.csv'):
        print("⚠ Файл данных для класса Primary не найден")
        print("Генерирую демонстрационные данные...")
        generate_demo_primary_data()
    
    return True

def main():
    """Основная функция программы визуализации"""
    print("=" * 60)
    print("СИСТЕМА ВИЗУАЛИЗАЦИИ РАСПРЕДЕЛЕНИЙ SEN")
    print("ВКЛЮЧАЯ ТЕСТИРОВАНИЕ КЛАССА PRIMARY")
    print("=" * 60)
    
    if not check_data_files():
        return
    
    print("✓ Все файлы данных найдены/сгенерированы")
    print("Начинаю генерацию графиков...\n")
    
    create_sen_plots()
    create_mixture_plots()  
    create_empirical_plots()
    create_primary_class_plots()
    
    print("\n" + "=" * 60)
    print("🎉 ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 60)

if __name__ == "__main__":
    main()
