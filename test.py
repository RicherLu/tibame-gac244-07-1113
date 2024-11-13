from sklearn import datasets
"""
這個程式使用 scikit-learn 庫來訓練和評估 K-最近鄰居（K-Nearest Neighbors, KNN）分類器。
步驟如下：
1. 從 scikit-learn 庫中載入 Iris 資料集。
2. 將資料集分割成訓練集和測試集。
3. 對特徵進行標準化處理。
4. 建立並訓練 KNN 分類器。
5. 在測試集上進行預測。
6. 計算分類器的準確率並輸出。
使用的主要函式和類別：
- datasets.load_iris: 載入 Iris 資料集。
- train_test_split: 將資料集分割成訓練集和測試集。
- StandardScaler: 對特徵進行標準化處理。
- KNeighborsClassifier: 建立 KNN 分類器。
- accuracy_score: 計算分類器的準確率。
變數：
- X: 特徵矩陣。
- y: 標籤向量。
- X_train: 訓練集的特徵矩陣。
- X_test: 測試集的特徵矩陣。
- y_train: 訓練集的標籤向量。
- y_test: 測試集的標籤向量。
- scaler: 標準化處理器。
- knn: KNN 分類器。
- y_pred: 測試集的預測標籤。
- accuracy: 分類器的準確率。
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 載入 Iris 資料集
print("載入 Iris 資料集...")
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 將資料集分割成訓練集和測試集
print("將資料集分割成訓練集和測試集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # 分割資料集，設置測試集佔 30%

# 對特徵進行標準化處理
print("對特徵進行標準化處理...")
scaler = StandardScaler()   # 創建標準化處理器實例
X_train = scaler.fit_transform(X_train) # 標準化訓練集特徵矩陣
X_test = scaler.transform(X_test)   # 標準化測試集特徵矩陣

# 建立並訓練 K-最近鄰居分類器
print("建立並訓練 K-最近鄰居分類器...")
knn = KNeighborsClassifier(n_neighbors=3)   # 創建 KNN 分類器實例，設置鄰居數為 3
knn.fit(X_train, y_train)   # 使用訓練集訓練 KNN 分類器

# 在測試集上進行預測
print("在測試集上進行預測...")
y_pred = knn.predict(X_test)    # 預測測試集標籤

# 計算分類器的準確率
print("計算分類器的準確率...")
accuracy = accuracy_score(y_test, y_pred)   # 計算準確率
print(f'Accuracy: {accuracy:.2f}')