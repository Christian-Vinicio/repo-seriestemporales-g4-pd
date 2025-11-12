"""
Transformadores personalizados para el pipeline de ventas
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Crea características temporales desde la fecha"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X['Fecha_Venta']):
            X['Fecha_Venta'] = pd.to_datetime(X['Fecha_Venta'], dayfirst=True, errors='coerce', format='mixed')
        
        X['anio'] = X['Fecha_Venta'].dt.year
        X['mes'] = X['Fecha_Venta'].dt.month
        X['dia_semana'] = X['Fecha_Venta'].dt.dayofweek
        X['dia_mes'] = X['Fecha_Venta'].dt.day
        X['trimestre'] = X['Fecha_Venta'].dt.quarter
        
        X['temporada'] = X['mes'].apply(lambda m: 
            'Invierno' if m in [12, 1, 2] else
            'Primavera' if m in [3, 4, 5] else
            'Verano' if m in [6, 7, 8] else 'Otonio'
        )
        
        # Eliminar Fecha_Venta inmediatamente después de crear las features
        X = X.drop('Fecha_Venta', axis=1)
        
        return X

class LagCreator(BaseEstimator, TransformerMixin):
    """Crea variables lag y rolling means"""
    def __init__(self, lags=[1, 7, 14, 30]):
        self.lags = lags
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for lag in self.lags:
            X[f'unidades_lag_{lag}'] = X.groupby(['Codigo_Sucursal', 'Codigo_Producto'])['Unidades_Vendidas'].shift(lag)
        
        X['unidades_rolling_7'] = X.groupby(['Codigo_Sucursal', 'Codigo_Producto'])['Unidades_Vendidas'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        X['unidades_rolling_30'] = X.groupby(['Codigo_Sucursal', 'Codigo_Producto'])['Unidades_Vendidas'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Codifica variables categóricas con LabelEncoder robusto"""
    def __init__(self):
        self.encoders = {}
        self.default_values = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        for col in ['Codigo_Sucursal', 'Codigo_Producto', 'temporada']:
            if col in X.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(X[col].astype(str))
                # Guardar el valor más frecuente como default
                self.default_values[col] = X[col].mode()[0] if len(X[col].mode()) > 0 else X[col].iloc[0]
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, encoder in self.encoders.items():
            if col in X.columns:
                # Manejar valores no vistos
                X_col = X[col].astype(str).copy()
                
                # Reemplazar valores no vistos con el valor default
                unseen_mask = ~X_col.isin(encoder.classes_)
                if unseen_mask.any():
                    print(f"Advertencia: {unseen_mask.sum()} valores no vistos en '{col}' - usando valor default")
                    X_col[unseen_mask] = str(self.default_values[col])
                
                X[f'{col}_encoded'] = encoder.transform(X_col)
        
        # Eliminar columnas originales después de codificar
        cols_to_drop = ['Codigo_Sucursal', 'Codigo_Producto', 'temporada']
        X = X.drop([col for col in cols_to_drop if col in X.columns], axis=1)
        
        return X

class OutlierTreatment(BaseEstimator, TransformerMixin):
    """Trata outliers usando winsorización con IQR"""
    def __init__(self, columns):
        self.columns = columns
        self.limits = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.limits[col] = {
                    'lower': Q1 - 1.5 * IQR,
                    'upper': Q3 + 1.5 * IQR
                }
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, limits in self.limits.items():
            if col in X.columns:
                X[f'{col}_tratado'] = X[col].clip(limits['lower'], limits['upper'])
                # Eliminar columna original
                X = X.drop(col, axis=1)
        return X

class LogTransformation(BaseEstimator, TransformerMixin):
    """Aplica transformación logarítmica"""
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[f'{col}_log'] = np.log1p(X[col])
        return X

class ToNumericTransformer(BaseEstimator, TransformerMixin):
    """Convierte DataFrame a array numpy eliminando columnas no numéricas"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Seleccionar solo columnas numéricas
            X_numeric = X.select_dtypes(include=[np.number])
            return X_numeric.values
        return X