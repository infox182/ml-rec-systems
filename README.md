# rec-systems
Проект в ИТМО по рекомендательным системам / инженерным практикам

## Настройка среды для запуска

#### Установка пакетного менеджера
```
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py
```

#### Развертывание окружения
```
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

#### Сборка пакета
```
  python3 -m build
  twine upload --repository testpypi dist/*
```

#### Ссылка на пакет в pypi-test
```
  https://test.pypi.org/project/test-rec-system
```

#### Установка пакета из pypi-test
```
  pip install -i https://test.pypi.org/simple/ test-rec-system==1.0.0
```

## Используемые линтеры и форматеры
### Форматер
  **black**

### Плагины для **flake8**
1. flake8-commas
2. flake8-bugbear
3. flake8-return
4. flake8-builtins
5. flake8-unused-arguments

## Версионирование данных

Используется Git LFS

**Команды:**

#### Установка Git LFS на mac
```
  brew install git-lfs
```

#### Отслеживание файлов
```
  git lfs track "*.joblib"
  git lfs track "*.csv"
```

### Добавление файлов

```
  git add data/
  git add best_models/
```
