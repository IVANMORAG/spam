import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# Ruta del dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'KDDTrain+.arff')


def load_kdd_dataset(data_path=None):
    """Lectura del DataSet NSL-KDD desde archivo ARFF"""
    if data_path is None:
        data_path = DATA_PATH
    
    try:
        import arff
        with open(data_path, 'r') as train_set:
            dataset = arff.load(train_set)
            attributes = [attr[0] for attr in dataset["attributes"]]
            return pd.DataFrame(dataset["data"], columns=attributes)
    except ImportError:
        # Fallback: leer como CSV si no está arff
        # El archivo .arff tiene una sección de datos que es CSV
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        # Encontrar donde empieza @DATA
        data_start = 0
        attributes = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line.lower().startswith('@attribute'):
                parts = line.split()
                attr_name = parts[1]
                attributes.append(attr_name)
            if line.lower() == '@data':
                data_start = i + 1
                break
        
        # Leer datos
        data_lines = [l.strip() for l in lines[data_start:] if l.strip() and not l.startswith('%')]
        data = [line.split(',') for line in data_lines]
        
        return pd.DataFrame(data, columns=attributes)


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """Divide el dataset en train, validation y test (60/20/20)"""
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    return train_set, val_set, test_set


# Pipeline numérico
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('rbst_scaler', RobustScaler()),
])


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """Transformador personalizado para One-Hot Encoding"""
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._columns = None

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._oh.fit(X_cat)
        self._columns = self._oh.get_feature_names_out(X_cat.columns)
        return self

    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        X_cat_oh = self._oh.transform(X_cat)
        return pd.DataFrame(X_cat_oh, columns=self._columns, index=X.index)


class DataFramePreparer(BaseEstimator, TransformerMixin):
    """Transformador que prepara todo el conjunto de datos"""
    def __init__(self):
        self._full_pipeline = None
        self._num_attribs = None
        self._cat_attribs = None

    def fit(self, X, y=None):
        self._num_attribs = list(X.select_dtypes(exclude=['object']).columns)
        self._cat_attribs = list(X.select_dtypes(include=['object']).columns)
        
        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, self._num_attribs),
            ("cat", CustomOneHotEncoder(), self._cat_attribs),
        ])
        self._full_pipeline.fit(X)
        return self

    def transform(self, X, y=None):
        return self._full_pipeline.transform(X)


# ============ FUNCIONES PARA LAS VISTAS ============

def get_dataset_info():
    """Obtiene información básica del dataset"""
    df = load_kdd_dataset()
    
    # Convertir tipos numéricos
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    
    info = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        'null_counts': df.isnull().sum().to_dict(),
    }
    return info, df


def get_visualization_data():
    """Obtiene datos para la página de visualización"""
    df = load_kdd_dataset()
    
    # Convertir tipos numéricos
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    
    # Head
    head_data = df.head(10).to_dict('records')
    head_columns = list(df.columns)
    
    # Describe
    describe_data = df.describe().round(2).to_dict()
    
    # Value counts
    class_counts = df['class'].value_counts().head(10).to_dict()
    protocol_counts = df['protocol_type'].value_counts().to_dict()
    service_counts = df['service'].value_counts().head(10).to_dict()
    flag_counts = df['flag'].value_counts().to_dict()
    
    # Info
    num_cols = len(df.select_dtypes(exclude=['object']).columns)
    cat_cols = len(df.select_dtypes(include=['object']).columns)
    
    return {
        'head_data': head_data,
        'head_columns': head_columns,
        'describe_data': describe_data,
        'class_counts': class_counts,
        'protocol_counts': protocol_counts,
        'service_counts': service_counts,
        'flag_counts': flag_counts,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'total_rows': len(df),
        'total_cols': len(df.columns),
    }


def get_division_data():
    """Obtiene datos para la página de división"""
    df = load_kdd_dataset()
    
    # Convertir tipos
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    
    # División estratificada
    train_set, val_set, test_set = train_val_test_split(df, stratify="protocol_type")
    
    # Distribuciones
    train_protocol = train_set['protocol_type'].value_counts().to_dict()
    val_protocol = val_set['protocol_type'].value_counts().to_dict()
    test_protocol = test_set['protocol_type'].value_counts().to_dict()
    
    # Calcular porcentajes
    train_pct = {k: round(v/len(train_set)*100, 1) for k, v in train_protocol.items()}
    val_pct = {k: round(v/len(val_set)*100, 1) for k, v in val_protocol.items()}
    test_pct = {k: round(v/len(test_set)*100, 1) for k, v in test_protocol.items()}
    
    return {
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set),
        'train_protocol': train_protocol,
        'val_protocol': val_protocol,
        'test_protocol': test_protocol,
        'train_pct': train_pct,
        'val_pct': val_pct,
        'test_pct': test_pct,
        'total_features': len(df.columns) - 1,  # -1 por la clase
    }


def get_preparation_data():
    """Obtiene datos para la página de preparación"""
    df = load_kdd_dataset()
    
    # Convertir tipos
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    
    num_attribs = list(df.select_dtypes(exclude=['object']).columns)
    cat_attribs = list(df.select_dtypes(include=['object']).columns)
    
    # Valores únicos de categóricos
    cat_unique = {col: df[col].nunique() for col in cat_attribs}
    cat_values = {col: df[col].unique().tolist()[:10] for col in cat_attribs}
    
    return {
        'num_attribs': num_attribs,
        'cat_attribs': cat_attribs,
        'num_count': len(num_attribs),
        'cat_count': len(cat_attribs),
        'cat_unique': cat_unique,
        'cat_values': cat_values,
    }


def get_evaluation_data():
    """Entrena el modelo y obtiene métricas de evaluación"""
    df = load_kdd_dataset()
    
    # Convertir tipos
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    
    # División
    train_set, val_set, test_set = train_val_test_split(df, stratify="protocol_type")
    
    # Separar X e y
    X_train = train_set.drop("class", axis=1)
    y_train = train_set["class"].copy()
    X_val = val_set.drop("class", axis=1)
    y_val = val_set["class"].copy()
    
    # Preparar datos
    preparer = DataFramePreparer()
    X_train_prep = preparer.fit_transform(X_train)
    X_val_prep = preparer.transform(X_val)
    
    # Entrenar modelo
    clf = LogisticRegression(max_iter=5000, n_jobs=-1)
    clf.fit(X_train_prep, y_train)
    
    # Predicciones
    y_pred = clf.predict(X_val_prep)
    
    # Métricas
    accuracy = accuracy_score(y_val, y_pred)
    
    # Classification report como dict
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    
    # Matriz de confusión
    cm = confusion_matrix(y_val, y_pred)
    labels = sorted(y_val.unique())
    
    # Convertir matriz a formato para gráfica
    cm_data = {
        'labels': labels[:10],  # Limitar a 10 clases para visualización
        'matrix': cm[:10, :10].tolist() if len(labels) > 10 else cm.tolist(),
    }
    
    # Top clases para el reporte
    top_classes = ['normal', 'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back']
    class_metrics = {}
    for cls in top_classes:
        if cls in report:
            class_metrics[cls] = {
                'precision': round(report[cls]['precision'], 2),
                'recall': round(report[cls]['recall'], 2),
                'f1': round(report[cls]['f1-score'], 2),
                'support': report[cls]['support'],
            }
    
    return {
        'accuracy': round(accuracy * 100, 1),
        'precision_macro': round(report['macro avg']['precision'] * 100, 1),
        'recall_macro': round(report['macro avg']['recall'] * 100, 1),
        'f1_macro': round(report['macro avg']['f1-score'] * 100, 1),
        'class_metrics': class_metrics,
        'cm_data': cm_data,
        'total_predictions': len(y_val),
    }


def check_dataset_exists():
    """Verifica si el dataset existe"""
    return os.path.exists(DATA_PATH)
