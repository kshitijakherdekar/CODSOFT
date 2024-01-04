#!/usr/bin/env python
# coding: utf-8

# ## Importing the packages

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# ## Importing the dataset

# In[2]:


titanic = pd.read_csv("C:/Users/kshit/Downloads/tested.csv")


# In[3]:


titanic.info()


# In[4]:


titanic.describe(include="all")


# In[5]:


titanic.isnull().sum()


# In[6]:


titanic.isnull().count()


# In[7]:


titanic.shape


# ## Calculating the mean

# In[8]:


Age= "Age"

mean_value = titanic[Age].mean()

titanic[Age].fillna(mean_value, inplace=True)


# In[9]:


titanic


# In[12]:


Fare= "Fare"

mean_value = titanic[Fare].mean()

titanic[Fare].fillna(mean_value, inplace=True)


# In[13]:


titanic


# In[14]:


titanic.isnull().sum()


# In[15]:


titanic.drop('Cabin', axis=1,inplace=True)


# In[16]:


titanic


# ## Count of Survived

# In[17]:


titanic['Survived'].value_counts()


# ## Create a scatterplot 

# In[18]:


import seaborn as sns


# In[19]:


sns.scatterplot(data=titanic, x='Age', y='Fare', hue='Survived')
plt.show()


# In[20]:


sns.scatterplot(data=titanic, x='Age', y='Fare', hue='Pclass')
plt.show()


# ## Replace 'male' with 1 and 'female' with 0 in the 'Sex' column

# In[21]:


titanic_update = pd.read_csv('C:/Users/kshit/Downloads/tested.csv')


# In[33]:


titanic_update['Sex'].replace({'male':1, 'female':0}, inplace=True)


# ## Replace 'Q' with 0, 'S' with 1, and 'C' with 2 in the 'Embarked' column

# In[34]:


titanic_update['Embarked'].replace({'Q':0, 'S':1, 'C':2}, inplace=True)
titanic_update.head()


# In[35]:


titanic_update = titanic.drop(['PassengerId','Name', 'Ticket'], axis=1)
titanic_update.head()


# ## Categorical variables to numerical representation

# In[36]:


numeric_features = ['Age', 'Fare']
categorical_features = ['Embarked', 'Sex', 'Pclass']


# In[37]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd


# In[38]:


label_encoder = LabelEncoder()
titanic['Sex'] = label_encoder.fit_transform(titanic['Sex'])
titanic['Embarked'] = label_encoder.fit_transform(titanic['Embarked'])


# ## Decision tree model

# In[39]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay


# In[40]:


X = titanic_update.drop('Survived', axis=1)
y = titanic_update['Survived']
X.head()


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


tree = DecisionTreeClassifier(random_state=42)


# In[43]:


tree.fit(X_train, y_train)


# In[44]:


y_pred_tree = tree.predict(X_test)


# In[45]:


print('Accuracy: ', accuracy_score(y_test, y_pred_tree))
print('Precision: ', precision_score(y_test, y_pred_tree))
print('Recall: ', recall_score(y_test, y_pred_tree))
print('F1 Score: ', f1_score(y_test, y_pred_tree))


# In[46]:


cm = confusion_matrix(y_test, y_pred_tree, labels=tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)
disp.plot();


# In[47]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)


# In[48]:


print('Accuracy: ', accuracy_score(y_test, y_pred_knn))
print('Precision: ', precision_score(y_test, y_pred_knn))
print('Recall: ', recall_score(y_test, y_pred_knn))
print('F1 Score: ', f1_score(y_test, y_pred_knn))


# In[49]:


cm = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot();

