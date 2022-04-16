# Ev Fiyat Tahmin Modeli
# İş Problemi

# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veri seti kullanılarak,farklı tipteki
# evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.

# Veri Seti Hikayesi

# Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor.
# Kaggle üzerinde bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz.
# Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır.
# Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri sizin  tahmin etmeniz beklenmektedir.

# Train veri seti için:

# Toplam Gözlem 1460
# Sayısal Değişken 38
# Kategorik Değişken 43



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" %x)
pd.set_option("display.width", 500)



def load():
    data = pd.read_csv("Dataset/train.csv")
    return data
def load_2():
    data = pd.read_csv("Dataset/test.csv")
    return data
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                       "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}), end="\n\n")
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # Nmerik görünülü kategorikleri çıkarttık
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def check_df(dataframe, head=10):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    # Grafikler birbirini ezmesin diye
    plt.show(block=True)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Elimizdeki veri seti üzerinden minimum hata ile ev fiyatlarını tahmin eden bir makine öğrenmesi modeli geliştiriniz.


# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Gelişmiş ağaç yöntemleri ile modelleme
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model


######################################################
# EDA
######################################################

df_train = load()
df_test = load_2()

# Train
df_train.head()
df_train.shape
df_train.info()
# test
df_test.shape


# Numerik ve kategorik değişkenleri yakalama
cat_cols, num_cols, cat_but_car = grab_col_names(df_train)
cat_cols_test, num_cols_test, cat_but_car_test = grab_col_names(df_test)

# Numerik ve kategorik değişken analizi
for col in cat_cols:
    cat_summary(df_train, col)

num_summary(df_train, num_cols)

"""
for col in df_train.columns:
    plot_numerical_col(df_train, col)

"""

# Numerik ve kategorik değişkenler ile hedef değişken incelenmesi

for col in num_cols:
    target_summary_with_num(df_train, "SalePrice", col)

for col in cat_cols:
    target_summary_with_cat(df_train, "SalePrice", col)


# Aykırı gözlem analizi

for i in num_cols:
    print(i,":",check_outlier(df_train, i))

for i in num_cols_test:
    print(i,":",check_outlier(df_test, i))

df_test.head()

# Eksik gözlem analizi

na_col = missing_values_table(df_train, na_name=True)
na_col_test = missing_values_table(df_test, na_name=True)

df_train.shape
df_test.shape

####################################################
#  2. Data Preprocessing & Feature Engineering
####################################################

# Aykırı değerler baskılandı(Train ve Test değerleri için)
for i in num_cols:
    replace_with_thresholds(df_train, i)
for i in num_cols_test:
    replace_with_thresholds(df_test, i)

# Train ve Test veri setlerini birleştirdikten sonra eksik verileri doldurmak
# hem pratiklik açısından hemde tutarlılık açısından daha iyi olacaktır

df = pd.concat((df_train.drop("SalePrice",axis=1), df_test), "index" )

df.shape
# Eksik değerleri doldurma
# Eksik değerler veri seti için anlam ifade ettiği için silinmesi yanlış sonuçlar elde etmemize neden
# Olabilir
# Havuz kalitesi
df["PoolQC"].fillna('Yok', inplace=True)
# MiscFeaturec ---> Diğer kategorilerde bulunmayan özellikler
df["MiscFeature"].fillna('None', inplace=True)
# Alley ---> Sokak girişi tipi
df["Alley"].fillna('None', inplace=True)
# Fence ---> Çit kalitesi
df["Fence"].fillna('None', inplace=True)
# FireplaceQu ---> Şömine kalitesi
df["FireplaceQu"].fillna('None', inplace=True)

# LotFrontage ---> median ile doldurabiliriz
df["LotFrontage"].fillna(df["LotFrontage"].median(), inplace=True)

# Evde bulunan garaj ile ilgili bilgileri içeren değişkenler
df["GarageType"].fillna('None', inplace=True)
df["GarageYrBlt"].fillna(0, inplace=True)
df["GarageFinish"].fillna('None', inplace=True)
df["GarageQual"].fillna('None', inplace=True)
df["GarageCond"].fillna('None', inplace=True)
df['GarageArea'].fillna(0,inplace=True)
df['GarageCars'].fillna(0,inplace=True)

# BsmtExposure ---> Yürüyüş veya bahçe katı bodrum duvarları
# BsmtFinType1 ---> bodrum kalitesi

df["BsmtExposure"].fillna('None', inplace=True)
df["BsmtFinType1"].fillna('None', inplace=True)
df["BsmtFinType2"].fillna('None', inplace=True)
df["BsmtQual"].fillna('None', inplace=True)
df["BsmtCond"].fillna('None', inplace=True)
df['BsmtFullBath'].fillna(0,inplace=True)
df['BsmtHalfBath'].fillna(0,inplace=True)
df['BsmtFinSF1'].fillna(0,inplace=True)
df['BsmtFinSF2'].fillna(0,inplace=True)
df['BsmtUnfSF'].fillna(0,inplace=True)
df['TotalBsmtSF'].fillna(0,inplace=True)


df["MasVnrType"].fillna('None', inplace=True)
df["MasVnrArea"].fillna(0, inplace=True)

df["SaleType"].value_counts()
df['Electrical'].fillna('None',inplace=True)
# MSZoning--->Genel imar sınıflandırması
df['MSZoning'].fillna('Not_Exist',inplace=True)

df['Utilities'].fillna('None',inplace=True)

# Functional ---> Ev işlevselliği değerlendirmesi
df['Functional'].fillna('Not_Exist',inplace=True)

# Exterior1st --_> Evdeki dış kaplama
df['Exterior1st'].fillna('Other',inplace=True)
df['Exterior2nd'].fillna('Other',inplace=True)
df['KitchenQual'].fillna('Other',inplace=True)
# SaleType ---> SaleType
df['SaleType'].fillna('Other',inplace=True)


missing_values_table(df)


################################
# Yeni değişken oluşturma...
################################
df_train.corr().sort_values("SalePrice", ascending=False)

#OverallQual ---Genel malzeme ve bitiş kalitesi
df["OverallQual"].value_counts()
#GrLivArea ---> Zemin oturma alanı metrekaresi
df["GrLivArea"].value_counts()
#GarageCars ---> garaj araç kapasitesi
df["GarageCars"].value_counts()
#TotalBsmtSF -- > Bodrum alanının toplam metre karesi
df["TotalBsmtSF"].value_counts()
#GarageArea ---> Garajın alanı
df["GarageArea"].value_counts()

df["Total_Area"] = df["GrLivArea"] + df["BsmtFinSF2"] + df["TotalBsmtSF"] + df["1stFlrSF"]+ df["2ndFlrSF"]
df["Overall"] = df["OverallQual"] * df["OverallCond"]
df["GarageCarArea"] = df["GarageArea"] / (df["GarageCars"]+0.1)

################################
# Encoding
################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if 23 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()

########################################################
# Model Kurma
########################################################

X_train = df.head(1460)
X_test = df.tail(1459)

y_Train = df_train["SalePrice"]
X_Train = X_train.drop(["Neighborhood", "Id"], axis=1)
X_Test = X_test.drop(["Neighborhood", "Id"], axis=1)

#############################################
# LinearRegression
#############################################
reg_model = LinearRegression().fit(X_Train, y_Train)
reg_model.intercept_
reg_model.coef_

y_pred = reg_model.predict(X_Train)

y_Train.mean()
#180450.73630136985

#MSE(Ortalama hatayı verir)
mean_squared_error(y_Train, y_pred)
# MSE = 364310978.64918274

#RMSE
np.sqrt(mean_squared_error(y_Train, y_pred))
# RMSE = 19086.932143463568

#MAE
mean_absolute_error(y_Train, y_pred)
# 12643.526591679181

# R-KARE
reg_model.score(X_Train, y_Train)
# 0.9382346465209339

################################################
# 3. Modeling using CART
################################################

cart_model = DecisionTreeRegressor(random_state=1)

cv_results = cross_validate(cart_model, X_Train, y_Train,
                            cv=5, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])

abs(cv_results['test_neg_root_mean_squared_error']).mean()
# RMSE = 39601.40253931294
cv_results['test_neg_mean_absolute_error'].mean()
# MAE = 25731.44623287671
cv_results['test_r2'].mean()
# 0.7320995213141505



################################################
# GBM (Gradient Boosting Machines)
################################################

gbm_model = GradientBoostingRegressor(random_state=17).fit(X_Train, y_Train)

cv_results = cross_validate(gbm_model, X_Train, y_Train,
                            cv=5, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])


cv_results['test_neg_root_mean_squared_error'].mean()
# RMSE = -24729.892994383503
cv_results['test_neg_mean_absolute_error'].mean()
# MAE = -15805.705715948705
cv_results['test_r2'].mean()
# 0.8957120338920477


################################################
# Random Forests
################################################

rf_model = RandomForestRegressor(random_state=17).fit(X_Train, y_Train)

cv_results = cross_validate(rf_model, X_Train, y_Train,
                            cv=5, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])


cv_results['test_neg_root_mean_squared_error'].mean()
# -26452.380957053647
cv_results['test_neg_mean_absolute_error'].mean()
# -16662.803825342464
cv_results['test_r2'].mean()
# 0.8802202600920378


################################################
# XGBoost
################################################

xgboost_model = XGBRegressor(random_state=17).fit(X_Train, y_Train)

cv_results = cross_validate(xgboost_model, X_Train, y_Train,
                            cv=5,scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])


cv_results['test_neg_root_mean_squared_error'].mean()
# -27553.37014274697
cv_results['test_neg_mean_absolute_error'].mean()
# -16977.590162136134
cv_results['test_r2'].mean()
# 0.8693283135580454



################################################
# LightGBM
################################################
lgbm_model = LGBMRegressor(random_state=17).fit(X_Train, y_Train)

cv_results = cross_validate(lgbm_model, X_Train, y_Train,
                            cv=5,scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])


cv_results['test_neg_root_mean_squared_error'].mean()
# -25502.87963290867
cv_results['test_neg_mean_absolute_error'].mean()
# -16119.207344510243
cv_results['test_r2'].mean()
# 0.8888530868287813

################################################
# CatBoost
################################################

catboost_model = CatBoostRegressor(random_state=17).fit(X_Train, y_Train)

cv_results = cross_validate(catboost_model, X_Train, y_Train,
                            cv=5,scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])


cv_results['test_neg_root_mean_squared_error'].mean()
# -23731.92420038664
cv_results['test_neg_mean_absolute_error'].mean()
# -14743.750400072804
cv_results['test_r2'].mean()
# 0.9038114203171425


#####################################################
# 4. Hyperparameter Optimization with GridSearchCV
#####################################################

################################################
# CatBoost
################################################
catboost_model = CatBoostRegressor(random_state=17)
catboost_model.get_params()
catboost_params = {"iterations": [200, 300, 500, 800],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X_Train, y_Train)
catboost_best_grid.best_params_

################################
# 5. Final Model
################################

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X_Train,y_Train)
cv_results = cross_validate(catboost_final, X_Train, y_Train,
                            cv=5, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])

cv_results['test_neg_root_mean_squared_error'].mean()
cv_results['test_neg_mean_absolute_error'].mean()
cv_results['test_r2'].mean()
