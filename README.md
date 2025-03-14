1. Была проведена обработка данных:
   - заполнены пропуски, удалены дубликаты, данные приведены к нужному типу
2. Выполнена визуализация и анализ корреляции признаков:
   - у целевой переменной selling_price:
      - сильная положительная корреляция с признакоми year, engine и max_power
      - сильная отрицательная с km_driven и mileage,
      - самая слабая корреляция с seats.
3. Обучение моделей
   - Были обучены модели линейной регрессии, Lasso регрессии c применением GridSearchCV, ElasticNet c применением GridSearchCV.
   - Лучший результат показала линейная регрессия.
   - Далее были закодированы категориальные фичи 'fuel', 'seller_type', 'transmission', 'owner', 'seats'
   - Применена модель Ridge для гребневой регрессии с применением GridSearchCV. Качество не превосходит линейную регрессию.
   - Реализована метрика business_metric, лучше всего решает задачу бизнеса модель линейной регрессии.
4. Реализован сервис на FastAPI:
   ![1](https://github.com/user-attachments/assets/952e6af5-bc0e-4270-b51a-fbcf75c11bfb)
![2](https://github.com/user-attachments/assets/d19b684a-fa3f-4724-811c-bce2ff2c3bb2)
![3](https://github.com/user-attachments/assets/4454ad91-c58a-4deb-8b9f-331fb826febc)
