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