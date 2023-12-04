#HURDAT Parsing -> Hurricane Objects w/ list of Entry objects
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


@dataclass
class Hurricane():
    name: str
    entries: list
    year: str
    basin: str
    atfc: int

    def __str__(self):
        return f'Storm {self.name} in {self.year} with {len(self.entries)} entries.'

    @classmethod
    def from_splt_hur(cls,splt_hur):
        splt_hur_head=[i.strip(' ') for i in splt_hur[0].split(',')]
        new = Hurricane(name=splt_hur_head[1],
                        year=int(splt_hur_head[0][-4:]),
                        basin=splt_hur_head[0][:2],
                        atfc=splt_hur_head[0][-6:-4],
                        entries=[])
        new.entries=Entry.entries_factory(splt_hur[1:])
        return new

    def total_max_wind(self):
        max=0
        for entry in self.entries:
            if entry.max_wind > max:
                max = entry.max_wind
        return max
    
    def max_min_pressure(self):
        max=0
        for entry in self.entries:
            if entry.min_pressure > max:
                max = entry.min_pressure
        return max

    def max_radius(self):
        max=0
        for entry in self.entries:
            if entry.radius > max:
                max = entry.radius
        return max
    
    def make_lf(self):
        lfs = []
        for entry in self.entries:
            if entry.identifier == 'L':
                lfs.append(entry)
        if lfs:
            return lfs
        else:
            return False

@dataclass
class Entry():
    date: datetime
    identifier: str
    status: str
    coordinates: tuple
    max_wind: int
    min_pressure: int
    radius: int

    def __str__(self):
        return f'{self.date} at {self.coordinates}\nID: {self.identifier}, Status: {self.status}\nMax Wind: {self.max_wind}, Min Pressure: {self.min_pressure}, Radius: {self.radius}'

    @classmethod
    def Factory(cls, entry):
        return Entry(date=datetime.strptime(f"{entry[0]}{entry[1]}",'%Y%m%d%H%M'),
                    identifier=entry[2],
                    status=entry[3],
                    coordinates=(entry[4], entry[5]),
                    max_wind=int(entry[6]),
                    min_pressure=int(entry[7]),
                    radius=int(entry[20]))

    @classmethod
    def entries_factory(cls,raw_entries):
        entries=[]
        for entry in raw_entries:
            entry_splt=[i.strip(' ') for i in entry.split(',')]
            entries.append(Entry.Factory(entry_splt))
        return entries

splt_hurricanes_raw=[]
with open("hurdat1923_2023.txt","r") as f:
    hur=[]
    for line in f:
        if line[0].isalpha():
            if hur != []:
                splt_hurricanes_raw.append(hur)
                hur=[]
        hur.append(line.strip())

hurricanes_classed=[]
for i in splt_hurricanes_raw:
    hurricanes_classed.append(Hurricane.from_splt_hur(i))

hur_cat = []
for i in hurricanes_classed:
    hur_cat_ind = [f"{i.name}{i.year}"]
    hur_cat_ind.append(i.entries[0].coordinates[0])
    hur_cat_ind.append(i.entries[0].coordinates[1])
    if i.make_lf():
        hur_cat_ind.append(1)
    else:
        hur_cat_ind.append(0)
    if i.total_max_wind() >= 96 and i.total_max_wind() <= 112:
        hur_cat_ind.append(1)
    else:
        hur_cat_ind.append(0)
    if i.total_max_wind() >= 113 and i.total_max_wind() <= 136:
        hur_cat_ind.append(1)
    else:
        hur_cat_ind.append(0)
    if i.total_max_wind() >= 137:
        hur_cat_ind.append(1)
    else:
        hur_cat_ind.append(0)
    hur_cat.append(hur_cat_ind)

df_hur_cat = pd.DataFrame(hur_cat,columns=['ID','Lat','Long','Landfall','Cat. 3','Cat. 4','Cat. 5'])

for i in list(itertools.combinations(['Cat. 3','Cat. 4','Cat. 5'],2)):
    print(i)
    print(ttest_ind(df_hur_cat[i[0]],df_hur_cat[i[1]]))

#True Lat/Long
df_hur_cat['Lat'] = df_hur_cat['Lat'].apply(lambda x:float(x[:-1]))
df_hur_cat['Long'] = df_hur_cat['Long'].apply(lambda x:float(x[:-1]))
x=np.asarray(df_hur_cat[['Lat','Long']])
x=preprocessing.StandardScaler().fit(x).transform(x)
y=np.asarray(df_hur_cat['Landfall'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

LR = LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
yhat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)
print(metrics.accuracy_score(y_test,yhat))
cm=confusion_matrix(y_test,yhat)
print(cm)

class_label = ["Positive", "Negative"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("True Lat/Long Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# # 8 & 20 Lat
df_hur_cat['Lat'] = df_hur_cat['Lat'].apply(lambda x:abs(float(x[:-1])-14))
df_hur_cat['Long'] = df_hur_cat['Long'].apply(lambda x:abs(float(x[:-1])-55.7))
x=np.asarray(df_hur_cat[['Lat','Long']])
x=preprocessing.StandardScaler().fit(x).transform(x)
y=np.asarray(df_hur_cat['Landfall'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

LR = LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
yhat = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)
print(metrics.accuracy_score(y_test,yhat))
cm=confusion_matrix(y_test,yhat)
print(cm)

class_label = ["Positive", "Negative"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Absolute Lat/Long Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# Category & Lat/Long
df_hur_cat['Lat'] = df_hur_cat['Lat'].apply(lambda x:float(x[:-1]))
df_hur_cat['Long'] = df_hur_cat['Long'].apply(lambda x:float(x[:-1]))
for i in ['Cat. 3','Cat. 4','Cat. 5']:
    x=np.asarray(df_hur_cat[['Lat','Long']])
    x=preprocessing.StandardScaler().fit(x).transform(x)
    y=np.asarray(df_hur_cat[i])

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    print ('Train set:', x_train.shape,  y_train.shape)
    print ('Test set:', x_test.shape,  y_test.shape)

    LR = LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
    yhat = LR.predict(x_test)
    yhat_prob = LR.predict_proba(x_test)
    print(i)
    print(metrics.accuracy_score(y_test,yhat))
    cm=confusion_matrix(y_test,yhat)
    print(cm)
    class_label = ["Positive", "Negative"]
    df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
    sns.heatmap(df_cm, annot = True, fmt = "d")
    plt.title(f"{i} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()