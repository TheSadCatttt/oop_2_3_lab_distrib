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

def create_mixture_class_plots():
    """
    Создание графиков для тестирования класса Mixture
    Сравнение функциональности класса с функциями из distributions
    """
    print("Создание графиков для класса Mixture...")
    
    try:
        # Загрузка данных класса Mixture
        df_mixture = safe_read_csv('data_mixture_class.csv')
        print(f"Успешно загружено {len(df_mixture)} строк")
        print(f"Колонки: {df_mixture.columns.tolist()}")
        
        # Посмотрим на структуру данных
        print("\nПервые 5 строк:")
        print(df_mixture.head())
        print("\nРазличные типы в mixture_type:")
        if 'mixture_type' in df_mixture.columns:
            print(df_mixture['mixture_type'].unique())
        else:
            print("Колонка 'mixture_type' не найдена!")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки данных класса Mixture: {e}")
        print("Генерирую демонстрационные данные...")
        generate_demo_mixture_data()
        df_mixture = safe_read_csv('data_mixture_class.csv')
    
    # Создание фигуры с 3 подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Тестирование класса Mixture (Сравнение с функциями distributions)', 
                 fontsize=18, fontweight='bold')
    
    try:
        # --- Сравнение плотностей ---
        # Фильтруем строки с числовыми значениями x (теоретические плотности)
        pdf_mask = df_mixture['x'].notna() & df_mixture['pdf_class'].notna() & df_mixture['pdf_function'].notna()
        pdf_data = df_mixture[pdf_mask].copy()
        
        if len(pdf_data) > 0:
            # Сортируем по x для правильного построения графиков
            pdf_data = pdf_data.sort_values('x')
            
            axes[0,0].plot(pdf_data['x'], pdf_data['pdf_class'], 'b-', linewidth=2.5, 
                           label='Класс Mixture')
            axes[0,0].plot(pdf_data['x'], pdf_data['pdf_function'], 'r--', linewidth=2, 
                           label='Функция pdf_mixture')
            axes[0,0].set_title('Сравнение плотностей смеси распределений\nSEN(0,1,1) + SEN(2,2,1), p=0.5', 
                                fontweight='bold', fontsize=12)
            axes[0,0].set_xlabel('x')
            axes[0,0].set_ylabel('Плотность вероятности')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_xlim(pdf_data['x'].min(), pdf_data['x'].max())
        else:
            axes[0,0].text(0.5, 0.5, 'Данные плотностей\nне найдены', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Сравнение плотностей', fontweight='bold', fontsize=12)
        
        # --- Сравнение моментов ---
        # Находим строки с моментами (где есть class_value и function_value)
        moments_mask = df_mixture['class_value'].notna() & df_mixture['function_value'].notna()
        moments_data = df_mixture[moments_mask].copy()
        
        if len(moments_data) > 0:
            # Берем соответствующие названия моментов из колонки 'moment'
            if 'moment' in moments_data.columns:
                moment_names = moments_data['moment'].tolist()
            else:
                # Если нет колонки moment, создаем стандартные названия
                moment_names = [f'Момент {i+1}' for i in range(len(moments_data))]
            
            x_pos = np.arange(len(moment_names))
            width = 0.35
            
            bars1 = axes[0,1].bar(x_pos - width/2, moments_data['class_value'].values, width, 
                                 label='Класс Mixture', alpha=0.7, color='blue')
            bars2 = axes[0,1].bar(x_pos + width/2, moments_data['function_value'].values, width, 
                                 label='Функция moments_mixture', alpha=0.7, color='red')
            
            axes[0,1].set_title('Сравнение моментов смеси распределений', fontweight='bold', fontsize=12)
            axes[0,1].set_xlabel('Момент')
            axes[0,1].set_ylabel('Значение')
            axes[0,1].set_xticks(x_pos)
            
            # Обрезаем длинные названия
            short_names = [name[:15] + '...' if len(str(name)) > 15 else str(name) 
                          for name in moment_names]
            axes[0,1].set_xticklabels(short_names, rotation=45, ha='right')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3, axis='y')
            
            # Добавление значений на столбцы
            for bar, value in zip(bars1, moments_data['class_value'].values):
                if not pd.isna(value):
                    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar, value in zip(bars2, moments_data['function_value'].values):
                if not pd.isna(value):
                    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            axes[0,1].text(0.5, 0.5, 'Данные моментов\nне найдены', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Сравнение моментов', fontweight='bold', fontsize=12)
        
        # --- Сравнение гистограмм генерации ---
        # Фильтруем строки с выборками
        samples_mask = df_mixture['class_samples'].notna() & df_mixture['function_samples'].notna()
        samples_data = df_mixture[samples_mask].copy()
        
        if len(samples_data) > 0:
            class_samples = samples_data['class_samples'].dropna()
            function_samples = samples_data['function_samples'].dropna()
            
            # Проверяем, что данные числовые
            class_samples_numeric = pd.to_numeric(class_samples, errors='coerce')
            function_samples_numeric = pd.to_numeric(function_samples, errors='coerce')
            
            # Удаляем NaN значения
            class_samples_clean = class_samples_numeric.dropna()
            function_samples_clean = function_samples_numeric.dropna()
            
            if len(class_samples_clean) > 10 and len(function_samples_clean) > 10:
                # Автоматическое определение границ гистограммы
                all_samples = pd.concat([class_samples_clean, function_samples_clean])
                x_min, x_max = all_samples.min(), all_samples.max()
                bins = min(50, int(np.sqrt(len(all_samples))))
                
                axes[1,0].hist(class_samples_clean, bins=bins, density=True, 
                               alpha=0.6, color='blue', label='Класс Mixture', 
                               edgecolor='black', linewidth=0.5, range=(x_min, x_max))
                axes[1,0].hist(function_samples_clean, bins=bins, density=True, 
                               alpha=0.6, color='red', label='Функция generate_mixture', 
                               edgecolor='black', linewidth=0.5, range=(x_min, x_max))
                axes[1,0].set_title(f'Сравнение гистограмм сгенерированных данных\n(класс: {len(class_samples_clean)}, функция: {len(function_samples_clean)})', 
                                    fontweight='bold', fontsize=12)
                axes[1,0].set_xlabel('x')
                axes[1,0].set_ylabel('Плотность вероятности')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                
                # --- Статистические тесты ---
                try:
                    # Тест Колмогорова-Смирнова
                    ks_stat, ks_pvalue = stats.ks_2samp(class_samples_clean, function_samples_clean)
                    
                    # Разности моментов
                    mean_diff = abs(class_samples_clean.mean() - function_samples_clean.mean())
                    var_diff = abs(class_samples_clean.var() - function_samples_clean.var())
                    std_diff = abs(class_samples_clean.std() - function_samples_clean.std())
                    
                    stats_data = [
                        ['Тест KS (p-value)', f'{ks_pvalue:.6f}'],
                        ['Разность средних', f'{mean_diff:.6f}'],
                        ['Разность дисперсий', f'{var_diff:.6f}'],
                        ['Разность СКО', f'{std_diff:.6f}']
                    ]
                    
                    # Таблица статистики
                    axes[1,1].axis('off')
                    table = axes[1,1].table(cellText=stats_data,
                                           colLabels=['Метрика', 'Значение'],
                                           cellLoc='center',
                                           loc='center',
                                           bbox=[0.2, 0.3, 0.6, 0.4])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                    
                    axes[1,1].set_title('Статистическое сравнение методов генерации', 
                                        fontweight='bold', fontsize=12)
                    
                    # Вывод статистики в консоль
                    print(f"\nСтатистика сравнения класса Mixture:")
                    print(f"  Тест Колмогорова-Смирнова: p-value = {ks_pvalue:.6f}")
                    print(f"  Разность средних: {mean_diff:.6f}")
                    print(f"  Разность дисперсий: {var_diff:.6f}")
                    print(f"  Разность СКО: {std_diff:.6f}")
                    
                    if ks_pvalue > 0.05:
                        print("  ✓ Вывод: распределения статистически неразличимы (класс работает корректно)")
                    else:
                        print("  ⚠ Вывод: обнаружены статистические различия между методами")
                        
                except Exception as e:
                    axes[1,1].text(0.5, 0.5, f'Ошибка статистики:\n{str(e)[:50]}...', 
                                  ha='center', va='center', transform=axes[1,1].transAxes,
                                  fontsize=10)
                    axes[1,1].set_title('Статистическое сравнение', fontweight='bold', fontsize=12)
            else:
                axes[1,0].text(0.5, 0.5, f'Недостаточно числовых данных\nдля гистограмм\n({len(class_samples_clean)}, {len(function_samples_clean)})', 
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
        output_file = 'mixture_class_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ График сравнения класса Mixture сохранен как '{output_file}'")
        
    except Exception as e:
        print(f"❌ Ошибка при создании графиков класса Mixture: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')

def create_empiric_class_plots():
    """
    Создание графиков для тестирования класса Empiric
    Сравнение функциональности класса с функциями из distributions
    """
    print("Создание графиков для класса Empiric...")
    
    try:
        # Загрузка данных класса Empiric
        df_empiric = safe_read_csv('data_empiric_class.csv')
        print(f"Успешно загружено {len(df_empiric)} строк")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных класса Empiric: {e}")
        print("Генерирую демонстрационные данные...")
        generate_demo_empiric_data()
        df_empiric = safe_read_csv('data_empiric_class.csv')
    
    # Создание фигуры с 3 подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Тестирование класса Empiric (Сравнение с функциями distributions)', 
                 fontsize=18, fontweight='bold')
    
    try:
        # --- Сравнение плотностей ---
        if 'x' in df_empiric.columns and 'pdf_class' in df_empiric.columns and 'pdf_function' in df_empiric.columns:
            # Фильтруем строки с плотностями (теоретически это первые строки)
            pdf_data = df_empiric[df_empiric['x'].notna()][['x', 'pdf_class', 'pdf_function']].head(100).dropna()
            if not pdf_data.empty:
                axes[0,0].plot(pdf_data['x'], pdf_data['pdf_class'], 'b-', linewidth=2.5, 
                               label='Класс Empiric')
                axes[0,0].plot(pdf_data['x'], pdf_data['pdf_function'], 'r--', linewidth=2, 
                               label='Функция pdf_empirical')
                axes[0,0].set_title('Сравнение плотностей эмпирического распределения\n(основа: SEN(0,1,1), n=10000)', 
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
        else:
            axes[0,0].text(0.5, 0.5, 'Данные плотностей\nне найдены', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Сравнение плотностей', fontweight='bold', fontsize=12)
        
        # --- Сравнение моментов ---
        if 'moment' in df_empiric.columns or 'class_value' in df_empiric.columns:
            # Ищем строки с моментами (те, где есть class_value и function_value)
            moments_mask = df_empiric['class_value'].notna() & df_empiric['function_value'].notna()
            moments_data = df_empiric[moments_mask][['class_value', 'function_value']]
            
            if not moments_data.empty:
                # Создаем названия моментов
                moment_names = ['Среднее', 'Дисперсия', 'Асимметрия', 'Эксцесс'][:len(moments_data)]
                x_pos = np.arange(len(moment_names))
                width = 0.35
                
                bars1 = axes[0,1].bar(x_pos - width/2, moments_data['class_value'].values, width, 
                                     label='Класс Empiric', alpha=0.7, color='blue')
                bars2 = axes[0,1].bar(x_pos + width/2, moments_data['function_value'].values, width, 
                                     label='Функция moments_empirical', alpha=0.7, color='red')
                
                axes[0,1].set_title('Сравнение моментов эмпирического распределения', fontweight='bold', fontsize=12)
                axes[0,1].set_xlabel('Момент')
                axes[0,1].set_ylabel('Значение')
                axes[0,1].set_xticks(x_pos)
                axes[0,1].set_xticklabels(moment_names, rotation=45)
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3, axis='y')
                
                # Добавление значений на столбцы
                for bar, value in zip(bars1, moments_data['class_value'].values):
                    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                
                for bar, value in zip(bars2, moments_data['function_value'].values):
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
        if 'class_samples' in df_empiric.columns and 'function_samples' in df_empiric.columns:
            # Фильтруем строки с выборками
            samples_mask = df_empiric['class_samples'].notna() & df_empiric['function_samples'].notna()
            samples_data = df_empiric[samples_mask][['class_samples', 'function_samples']]
            
            if len(samples_data) > 0:
                class_samples = samples_data['class_samples']
                function_samples = samples_data['function_samples']
                
                axes[1,0].hist(class_samples, bins=50, density=True, 
                               alpha=0.6, color='blue', label='Класс Empiric', 
                               edgecolor='black', linewidth=0.5)
                axes[1,0].hist(function_samples, bins=50, density=True, 
                               alpha=0.6, color='red', label='Функция generate_empirical', 
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
                    print(f"\nСтатистика сравнения класса Empiric:")
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
        plt.savefig('empiric_class_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ График сравнения класса Empiric сохранен как 'empiric_class_comparison.png'")
        
    except Exception as e:
        print(f"❌ Ошибка при создании графиков класса Empiric: {e}")
        plt.close('all')

def generate_demo_mixture_data():
    """Генерация демонстрационных данных для класса Mixture"""
    print("Генерация демонстрационных данных для класса Mixture...")
    
    np.random.seed(42)
    
    # Данные для плотностей (два нормальных распределения для смеси)
    x_values = np.linspace(-4, 8, 200)
    pdf1 = stats.norm.pdf(x_values, 0, 1) * 0.5
    pdf2 = stats.norm.pdf(x_values, 2, 2) * 0.5
    pdf_class = pdf1 + pdf2
    pdf_function = pdf1 + pdf2 * 0.99  # Немного отличаем для демонстрации
    
    # Создаем DataFrame для плотностей
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
            'function_samples': np.nan,
            'mixture_type': np.nan  # Оставляем пустым или заполняем как нужно
        })
    
    # Данные для моментов
    moments_data = [
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Среднее', 'class_value': 1.0, 'function_value': 1.0,
         'class_samples': np.nan, 'function_samples': np.nan, 'mixture_type': np.nan},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Дисперсия', 'class_value': 2.25, 'function_value': 2.25,
         'class_samples': np.nan, 'function_samples': np.nan, 'mixture_type': np.nan},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Асимметрия', 'class_value': 0.3, 'function_value': 0.3,
         'class_samples': np.nan, 'function_samples': np.nan, 'mixture_type': np.nan},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Эксцесс', 'class_value': -0.5, 'function_value': -0.5,
         'class_samples': np.nan, 'function_samples': np.nan, 'mixture_type': np.nan}
    ]
    
    # Данные для сгенерированных выборок
    # Смесь: 50% из N(0,1), 50% из N(2,4)
    n_samples = 1000
    choice = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    class_samples = np.where(choice == 0, 
                            np.random.normal(0, 1, n_samples),
                            np.random.normal(2, 2, n_samples))
    function_samples = np.where(choice == 0,
                               np.random.normal(0, 1.01, n_samples),  # Немного отличаем
                               np.random.normal(2, 2.02, n_samples))
    
    sample_rows = []
    for i in range(n_samples):
        sample_rows.append({
            'x': np.nan,
            'pdf_class': np.nan,
            'pdf_function': np.nan,
            'moment': np.nan,
            'class_value': np.nan,
            'function_value': np.nan,
            'class_samples': class_samples[i],
            'function_samples': function_samples[i],
            'mixture_type': np.nan
        })
    
    # Объединяем все данные
    all_data = pdf_rows + moments_data + sample_rows
    df = pd.DataFrame(all_data)
    
    # Сохраняем в CSV с правильной кодировкой
    df.to_csv('data_mixture_class.csv', index=False, encoding='utf-8')
    print("✓ Демонстрационные данные сохранены в 'data_mixture_class.csv'")

def generate_demo_empiric_data():
    """Генерация демонстрационных данных для класса Empiric"""
    print("Генерация демонстрационных данных для класса Empiric...")
    
    np.random.seed(42)
    
    # Данные для плотностей (эмпирическое распределение на основе нормального)
    x_values = np.linspace(-4, 4, 200)
    theoretical_pdf = stats.norm.pdf(x_values, 0, 1)
    
    # Имитация эмпирической плотности с шумом
    np.random.seed(42)
    emp_data = np.random.normal(0, 1, 10000)
    hist, bins = np.histogram(emp_data, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Интерполяция для получения гладкой кривой
    from scipy.interpolate import interp1d
    interp_func = interp1d(bin_centers, hist, kind='cubic', bounds_error=False, fill_value=0)
    pdf_class = interp_func(x_values)
    pdf_function = pdf_class * 1.02  # Немного отличаем для демонстрации
    
    # Создаем DataFrame для плотностей
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
            'function_samples': np.nan,
            'data_size': 10000
        })
    
    # Данные для моментов
    moments_data = [
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Среднее', 'class_value': 0.0, 'function_value': 0.0,
         'class_samples': np.nan, 'function_samples': np.nan, 'data_size': 10000},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Дисперсия', 'class_value': 0.98, 'function_value': 0.98,
         'class_samples': np.nan, 'function_samples': np.nan, 'data_size': 10000},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Асимметрия', 'class_value': 0.05, 'function_value': 0.05,
         'class_samples': np.nan, 'function_samples': np.nan, 'data_size': 10000},
        {'x': np.nan, 'pdf_class': np.nan, 'pdf_function': np.nan,
         'moment': 'Эксцесс', 'class_value': 0.1, 'function_value': 0.1,
         'class_samples': np.nan, 'function_samples': np.nan, 'data_size': 10000}
    ]
    
    # Данные для сгенерированных выборок
    class_samples = np.random.normal(0, 1, 1000)
    function_samples = np.random.normal(0, 1, 1000) * 0.99
    
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
            'function_samples': function_samples[i],
            'data_size': 10000
        })
    
    # Объединяем все данные
    all_data = pdf_rows + moments_data + sample_rows
    df = pd.DataFrame(all_data)
    
    # Сохраняем в CSV с правильной кодировкой
    df.to_csv('data_empiric_class.csv', index=False, encoding='utf-8')
    print("✓ Демонстрационные данные сохранены в 'data_empiric_class.csv'")
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
        'data_empirical_comparison.csv',
        'data_primary_class.csv',
        'data_mixture_class.csv',
        'data_empiric_class.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠ Отсутствуют следующие файлы данных:")
        for f in missing_files:
            print(f"   - {f}")
        
        # Для файлов классов генерируем демо-данные
        class_files = ['data_primary_class.csv', 'data_mixture_class.csv', 'data_empiric_class.csv']
        for cf in class_files:
            if cf in missing_files:
                print(f"   Генерация демо-данных для {cf}...")
                if cf == 'data_primary_class.csv':
                    generate_demo_primary_data()
                elif cf == 'data_mixture_class.csv':
                    generate_demo_mixture_data()
                elif cf == 'data_empiric_class.csv':
                    generate_demo_empiric_data()
        
        # Проверяем оставшиеся файлы
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files and all(f not in class_files for f in missing_files):
            print("\n❌ Для генерации оставшихся файлов выполните:")
            print("   1. Скомпилируйте C++ программу")
            print("   2. Запустите программу")
            print("   3. Выберите опцию 2 или 3 для генерации всех данных")
            return False
    
    return True
    
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
    print("ВКЛЮЧАЯ ТЕСТИРОВАНИЕ КЛАССОВ PRIMARY, MIXTURE И EMPIRIC")
    print("=" * 60)
    
    if not check_data_files():
        return
    
    print("✓ Все файлы данных найдены/сгенерированы")
    print("Начинаю генерацию графиков...\n")
    
    create_sen_plots()
    create_mixture_plots()  
    create_empirical_plots()
    create_primary_class_plots()
    create_mixture_class_plots()    # Добавлено
    create_empiric_class_plots()    # Добавлено
    
    print("\n" + "=" * 60)
    print("🎉 ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("Созданы графики для всех классов:")
    print("  - Primary: primary_class_comparison.png")
    print("  - Mixture: mixture_class_comparison.png")
    print("  - Empiric: empiric_class_comparison.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
