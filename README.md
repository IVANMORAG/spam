# NSL-KDD Network Intrusion Detection - Dashboard

Dashboard web para visualizar el análisis de Machine Learning del dataset NSL-KDD para detección de intrusiones de red.

## Requisitos Previos

- Python 3.10+
- pip
- Archivo `KDDTrain+.arff` del dataset NSL-KDD

## Instalación Local

### 1. Clonar/Descargar el proyecto

```bash
cd nsl_kdd_project
```

### 2. Descargar el Dataset

Descarga el dataset NSL-KDD desde: https://www.unb.ca/cic/datasets/nsl.html

Coloca el archivo `KDDTrain+.arff` en la carpeta `data/`:

```
nsl_kdd_project/
├── data/
│   └── KDDTrain+.arff   <-- AQUÍ
├── dashboard/
├── core/
...
```

### 3. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar migraciones

```bash
python manage.py migrate
```

### 6. Ejecutar servidor de desarrollo

```bash
python manage.py runserver
```

Acceder a: http://127.0.0.1:8000

---

## Despliegue en Render

### 1. Subir código a GitHub (incluyendo el dataset)

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

**Nota:** El archivo .arff debe estar en `data/KDDTrain+.arff`

### 2. Crear Web Service en Render

1. Click en **"New +"** → **"Web Service"**
2. Conectar repositorio de GitHub
3. Configurar:
   - **Name**: nsl-kdd-dashboard
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn core.wsgi:application`

### 3. Variables de Entorno

| Key | Value |
|-----|-------|
| `SECRET_KEY` | (generar una clave) |
| `DEBUG` | `False` |
| `PYTHON_VERSION` | `3.11.0` |

---

## Estructura del Proyecto

```
nsl_kdd_project/
├── core/                   # Configuración Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── dashboard/              # App principal
│   ├── templates/dashboard/
│   ├── templatetags/       # Filtros personalizados
│   ├── utils.py            # Funciones de ML
│   ├── views.py
│   └── urls.py
├── data/
│   └── KDDTrain+.arff      # Dataset (REQUERIDO)
├── static/css/
├── manage.py
├── requirements.txt
└── README.md
```

---

## Secciones del Dashboard

1. **Introducción**: Descripción del problema y dataset
2. **Visualización**: df.head(), value_counts, gráficas
3. **División**: Train/Val/Test split estratificado
4. **Preparación**: Transformaciones, atributos
5. **Pipelines**: Código de transformadores
6. **Evaluación**: Accuracy, métricas, classification report

---

## Tecnologías

- **Backend**: Django 4.2, scikit-learn, pandas
- **Frontend**: HTML5, Bootstrap 5, Chart.js
- **ML**: LogisticRegression, RobustScaler, OneHotEncoder

---

## Autor

Ivan - Proyecto de Machine Learning
