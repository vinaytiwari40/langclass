#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import string
import re
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# In[2]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# In[3]:


df1 = pd.read_csv('C:/Users/raish/OneDrive/Desktop/Language Detection.csv')
df2 = pd.read_csv('C:/Users/raish/OneDrive/Desktop/hindi.csv')

# In[4]:


df1

# In[5]:


df2

# In[6]:


df = df1.append(df2, ignore_index=True)

# In[8]:


df.info()

# In[9]:


df.Language.value_counts()

# In[10]:


df[df.Language == 'Russian'].sample(2)

# In[11]:


df[df.Language == 'Malayalam'].sample(2)

# In[12]:


df[df.Language == 'Arabic'].sample(2)

# In[13]:


df[df.Language == 'Tamil'].sample(2)

# In[14]:


df[df.Language == 'Kannada'].sample(2)


# In[15]:


# In[17]:


def removeSymbolsAndNumbers(text):
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[@]', '', text)

    return text.lower()


# In[18]:


def removeEnglishLetters(text):
    text = re.sub(r'[a-zA-Z]+', '', text)
    return text.lower()


# In[19]:


X0 = df.apply(
    lambda x: removeEnglishLetters(x.Text) if x.Language in ['Russian', 'Malyalam', 'Hindi', 'Kannada', 'Tamil',
                                                             'Arabic'] else x.Text, axis=1)
X0

# In[20]:


X1 = X0.apply(removeSymbolsAndNumbers)
X1

# In[21]:


y = df['Language']

# In[22]:


x_train, x_test, y_train, y_test = train_test_split(X1, y, random_state=42)

# In[23]:


vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char')

# In[24]:


model = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', LogisticRegression())
])

# In[25]:


model.fit(x_train, y_train)

# In[27]:


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# In[28]:


print("Accuracy is :", accuracy)

# In[29]:


print(classification_report(y_test, y_pred))


# In[30]:


def predict(text):
    lang = model.predict([text])
    print('The Language is in', lang[0])


# In[31]:


# English
predict("LANGUAGE DETECTION MODEL CHECK")
# French
predict("VÉRIFICATION DU MODÈLE DE DÉTECTION DE LA LANGUE")
# Arabic
predict("توففحص نموذج الكشف عن اللغة")
# Spanish
predict("VERIFICACIÓN DEL MODELO DE DETECCIÓN DE IDIOMAS")
# Malayalam
predict("ലാംഗ്വേജ് ഡിറ്റക്ഷൻ മോഡൽ ചെക്ക്")
# Russian
predict("ПРОВЕРКА МОДЕЛИ ОПРЕДЕЛЕНИЯ ЯЗЫКА")
# Hindi
predict('भाषा का पता लगाने वाले मॉडल की जांच')
# Hindi
predict(' boyit9h एनालिटिक्स alhgserog 90980879809 bguytfivb ahgseporiga प्रदान करता है')

# In[63]:


model.predict(['WASHINGTON: Le président Joe Biden a déclaré jeudi que '])[0]

# In[59]:


import pickle

# In[61]:


filename = 'ClassPick'
outfile = open(filename, 'wb')

# In[64]:


pickle.dump(model, outfile)
outfile.close()
