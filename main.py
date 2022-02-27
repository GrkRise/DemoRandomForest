
import numpy as np #библиотека NumPy
import pandas as pd #библиотека Pandas
import matplotlib.pyplot as plt #библиотека MatPlotLib
from sklearn.model_selection import train_test_split, learning_curve #библиотека scikit-learn
from sklearn.preprocessing import MinMaxScaler #библиотека scikit-learn
from sklearn.ensemble import RandomForestClassifier #библиотека scikit-learn
from sklearn import tree
import seaborn as sns

# https://www.kaggle.com/yasserh/wine-quality-dataset - dataset

df = pd.read_csv("WineQT.csv")
print(df)
print("\n")

df.drop("Id", axis=1, inplace = True)
print(df.describe())
print("\n")

print(df.quality.unique())
print("\n")

x = df.loc[:,df.columns!="quality"]
print(x)
print("\n")

y = df.loc[:,df.columns=="quality"]
print(y)
print("\n")

scaler = MinMaxScaler(feature_range = (0 , 1))
x_scaled = scaler.fit_transform(x)
print(x_scaled)
print("\n")

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

random_forest = RandomForestClassifier(
    n_estimators = 3,
    max_depth = 3,
    criterion="gini",
    max_features=2,
    min_samples_split=3,
    min_samples_leaf=3,
    random_state = 0)

random_forest.fit(x_train, y_train.values.ravel())

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=400)
for i in range(0, len(random_forest)):
    tree.plot_tree(random_forest[i],
                   feature_names=x.columns.array,
                   filled=True
                   )
    fig.savefig("image" + str(i) + ".png")
    plt.show()

print(random_forest.score(x_train, y_train))
print("\n")

print(random_forest.score(x_test, y_test))
print("\n")


pred_imp = pd.Series(random_forest.feature_importances_, index=x.columns.array).sort_values(ascending=False)
print(pred_imp)
print("\n")

sns.barplot(x=pred_imp, y=pred_imp.index)
plt.xlabel('Важность признаков')
plt.ylabel('Признаки')
plt.title('Визуализация важных признаков')
plt.savefig("importance_features.png")
plt.show()

pr = random_forest.predict(x_test)
print(pr)
print("\n")

print(pd.DataFrame(pr).describe())
print("\n")

print(pd.DataFrame(y_test.values).describe())
print("\n")

train_size, train_scores, test_scores = learning_curve(random_forest,
                                                       x_train,
                                                       y_train.values.ravel(),
                                                       train_sizes = np.arange(0.1,1.,0.2),
                                                       cv=3, scoring="accuracy")

print(train_size)
print("\n")

print(train_scores.mean(axis = 1))
print("\n")

print(test_scores.mean(axis = 1))
print("\n")

plt.grid(True)
plt.plot(train_size, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
plt.plot(train_size, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
plt.ylim(0.0, 1.1)
plt.legend(loc = 'lower right')
plt.savefig("learning_curve.png")
plt.show()





