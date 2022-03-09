# auto binary classifier

Библиотека **Авто ML** предназначена для автоматического выбора модели для бинарной классификации.

Осноным является класс AutoBinaryClassifier. При создании экземпляра AutoBinaryClassifier можно указать метрику для оценки моделей, а так же алгоритм валидации ('train/test-split', 'stratified_k_fold').
Так-же библиотеке есть класс Validator который при инициализации может принимать callable объекты или функции, которые должны вызываться с паттерном ```
callabel(X, y, classifier, get_metric(self.metric))``` 

Класс поддерживает методы fit/predict.

fit(X, y) - обучает и оценивает модели
predict(X) - Применяет наилучшую модель к X

## Установка зависимостей

```shell
pip install -r requiments.txt 
```

## Использование 

скопировать папку auto_binary_classificator в проект или в 'python_directory\Lib\site-packages'