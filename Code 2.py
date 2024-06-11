import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

Laptop_df = pd.read_csv("laptop_price.csv", encoding="latin1")
print(Laptop_df.shape)

class data_cleaning(BaseEstimator, TransformerMixin):
    @staticmethod
    def TB_to_GB(value):
        if str(value)[-2:] == 'TB':
            return_value = str(float(str(value)[:-2]) * 1024) + "GB"
            return return_value
        else:
            return str(value)
    @staticmethod
    def screen_type(value):
        if value[0] in ["1440x900", "1366x768", "1600x900", "1920x1080", "1920x1080", "2560x1440"]:
            return "Plain"
        elif value[0] == "Touchscreen":
            return "Touchscreen"
        else:
            return " ".join(value[0: 2])    
    def fit(self, X_df, y=None):
        return self
    def transform(self, X_df, y=None):
        X_df.drop(["Product"], axis=1, inplace=True)
        X_df["Ram"] = X_df["Ram"].str[:-2].astype("int64")
        X_df["Weight"] = X_df["Weight"].str[:-2].astype("float")
        X_df["Memory Type"] = X_df["Memory"].str.split(" ").apply(lambda x: x[1])
        X_df["Memory"] = X_df["Memory"].str.split(" ").apply(lambda x: x[0])
        X_df["Memory"] = X_df["Memory"].apply(data_cleaning.TB_to_GB).str[:-2].astype("float")
        X_df["Screen"] = X_df["ScreenResolution"].str.split(" ").apply(lambda x: x[-1])
        X_df["Screen Width"] = X_df["Screen"].str.split("x").apply(lambda x: x[0]).astype("int64")
        X_df["Screen Height"] = X_df["Screen"].str.split("x").apply(lambda x: x[-1]).astype("int64")
        X_df.drop(["Screen"], axis=1, inplace=True)
        X_df["Screen Type"] = X_df["ScreenResolution"].str.split(" ").apply(data_cleaning.screen_type)
        X_df["CPU Clock Speed"] = X_df["Cpu"].str.split(" ").apply(lambda x: x[-1]).str[:-3].astype("float")
        X_df["GPU Type"] = X_df["Gpu"].str.split(" ").apply(lambda x: x[0])
        X_df["CPU Type"] = X_df["Cpu"].str.split(" ").apply(lambda x: x[0])
        X_df.drop(columns=["ScreenResolution", "Cpu", "Gpu"], inplace=True)
        return X_df


Cleaner = data_cleaning()
Cleaned_df = Cleaner.fit_transform(Laptop_df)
print(Cleaned_df.info())
X = np.array(Cleaned_df.copy().drop(columns=["Price_euros"]))
y = np.array(np.log(Cleaned_df["Price_euros"]))
### log function is applied to prices as they were skewed, apply exp function to the predictions to show actual prices 
#print(X[0: 2, :])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
OHE = OneHotEncoder(drop="first", sparse=False)
Columns_trans = ColumnTransformer(remainder="passthrough", transformers=[
    ("Company", OHE, [1]),
    ("TypeName", OHE, [2]),
    ("Opsys", OHE, [6]),
    ("Memory Type", OHE, [8]),
    ("Screen Type", OHE, [11]),
    ("GPU Type", OHE, [13]),
    ("CPU Type", OHE, [14])
])
imputer = SimpleImputer(strategy="most_frequent")
standard = StandardScaler()
Reg = Ridge(alpha=10)
Reg2 = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
#model = GridSearchCV(Reg, cv=5, scoring="neg_mean_squared_error")

pipeline = Pipeline([
    ("Imputer", imputer),
    ("Column Transformer", Columns_trans),
    ("ML Algorithm", Reg2)
]
)

pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)
error = mean_absolute_error(y_test, y_predicted)
print(error)




