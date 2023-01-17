#!/usr/bin/env python
# coding: utf-8

# # Gini Index

# ### Importing Dependencies

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# ### Importing Dataset

# In[25]:


df = pd.read_csv(r'classification.csv')


# In[26]:


df


# ### Data Analysis

# In[27]:


df.info()


# In[28]:


df.isnull().sum()


# ### Data Visualisation

# In[29]:


df.Age.plot(kind='kde')


# In[30]:


df.EstimatedSalary.plot(kind='kde')


# ## Gini Index

# In[31]:


def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


# In[32]:


print(f"Gini Index of Age column is {gini(df['Age'])}")
print(f"Gini Index of Age column is {gini(df['EstimatedSalary'])}")


# # Decision Tree Classifier

# ### Importing Dataset

# In[33]:


df1 = pd.read_csv(r"diabetes.csv")


# In[34]:


df1.drop("Age",inplace=True,axis=1)


# In[35]:


df1


# ### Data Analysis

# In[36]:


df1.info()


# In[37]:


df1.isna().sum()


# ### Data Visualisation

# In[38]:


for i in df1:
    sns.histplot(df1[i])
    plt.show()


# ### Encoding

# In[39]:


gender_label = {'Male': 0, 'Female':1}
label_rest = {'No':0,'Yes': 1}
class_label = {'Positive': 1, 'Negative': 0}


# In[40]:


for i in df1:
    if i == 'Gender':
        df1[i] = df1[i].map(gender_label)
    elif i == "class":
        df1[i] = df1[i].map(class_label)
    else:
        df1[i] = df1[i].map(label_rest)


# In[41]:


x = df1.iloc[:, :-1].values
y = df1.iloc[:, -1].values


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)


# ### Model

# In[43]:


class Node:
    def __init__(self, feature = None, threshold  = None, df_left = None, df_right = None, gain = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.df_left = df_left
        self.df_right = df_right
        self.gain = gain
        self.value = value


# In[44]:


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    @staticmethod
    def _entropy(s):
        counts = np.bincount(np.array(s, dtype=np.int64))
        percentages = counts / len(s)

        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy
    
    def _information_gain(self, parent, left_child, right_child):
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    
    def _best_split(self, X, y):

        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            for threshold in np.unique(X_curr):
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]

                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
        n_rows, n_cols = X.shape
        
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            best = self._best_split(X, y)
            if best['gain'] > 0:
                left = self._build(
                    X=best['df_left'][:, :-1], 
                    y=best['df_left'][:, -1], 
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1], 
                    y=best['df_right'][:, -1], 
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    df_left=left, 
                    df_right=right, 
                    gain=best['gain']
                )
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):
        self.root = self._build(X, y)
        
    def _predict(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.df_left)
        
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.df_right)
        
    def predict(self, X):
        return [self._predict(x, self.root) for x in X]


# In[45]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[46]:


model = DecisionTree()
model.fit(x_train, y_train)
pred = model.predict(x_test)


# ### Checking Metrics

# In[47]:


print(accuracy_score(pred,y_test))


# In[48]:


print(confusion_matrix(pred,y_test))


# In[49]:


print(classification_report(pred, y_test))


# ## Decision Tree Using SK-Learn Libraries

# In[50]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# ### Checking Metrics

# In[51]:


print(accuracy_score(y_pred,y_test))


# In[52]:


print(classification_report(y_pred,y_test))


# In[53]:


print(confusion_matrix(y_test, y_pred))

