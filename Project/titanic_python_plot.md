```
# Imports
import math
# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go

from mpl_toolkits.basemap import Basemap

sns.set_style('whitegrid')

# %matplotlib inline

# # machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("/Users/JAY/Desktop/Udacity/Project2/Project/titanic_data.csv", dtype={"Age": np.float64}, )
# test_df = pd.read_csv("/Users/JAY/Desktop/Udacity/Project2/Project/Titanic_training.csv", dtype={"Age": np.float64}, )



################################################################################

# preview the data

# titanic_df.info()
# print("----------------------------")
# test_df.info()

# drop unnecessary columns, these columns won't be useful in analysis and prediction
# titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
# test_df    = test_df.drop(['Name','Ticket'], axis=1)

Emb = titanic_df['Embarked']

S_embark = []
C_embark = []
Q_embark = []

for embark in titanic_df['Embarked']:
    if embark == 'S':
        S_embark.append(embark)

    elif embark == 'C':
        C_embark.append(embark)

    elif embark =='Q':
        Q_embark.append(embark)

    else:
        print embark

# Embarked

################################added data clean process##########################################
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_df.drop("Ticket", axis=1, inplace=True)
titanic_df.drop("Cabin",axis=1,inplace=True)
titanic_df.drop("Fare", axis=1, inplace=True)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
##################################################################################################

print titanic_df.head()
#titanic data Pclass number to string

titanic_df['Pclass'] = titanic_df['Pclass'].astype(basestring)


# print titanic_df['Pclass']

# print titanic_df['Name']
# print titanic_df['Pclass']

plt.figure(1, figsize=(15,5))

# plot

plt.subplot(1,3,1)
# Present city by Class [hue should be data type string, object something like]
sns.countplot(x='Survived',data=titanic_df, hue='Pclass')

plt.title('Survived vs Pclass')
plt.xlabel('Survived')
plt.ylabel('Frequency of Pclass')
plt.ylim(0,500)
# plt.savefig('Survived_VS_Pclass.png')

plt.subplot(1,3,2)

sns.countplot(x='Survived', data=titanic_df, hue='Embarked')
plt.title('Survived vs Embarked')
plt.xlabel('Survived')
plt.ylabel('Frequency of Embarked')
plt.ylim(0,500)


plt.subplot(1,3,3)

sns.countplot(x='Survived', data=titanic_df, hue='Sex')
plt.title('Survived vs Sex')
plt.xlabel('Survived')
plt.ylabel('Frequency of Sex')
plt.ylim(0,500)


# n_bins = [min(titanic_df['Age'],max(titanic_df['Age']),1)]


############################################### Age vs Dead ################################


Age_mean = titanic_df['Age'].mean()

Re_age = titanic_df['Age'].fillna(Age_mean)
#
# print Re_age

Re_Age = []

for n in Re_age:
    Re_Age.append(int(n))

#print titanic_df['Survived']
# print Re_Age

# print len(Re_Age)
# print Re_Age[0:5]

Dead_idx = titanic_df[titanic_df['Survived']==0].index.tolist()
Survive_idx = titanic_df[titanic_df['Survived']==1].index.tolist()

Re_Survived = []
Re_Dead = []

for i in Dead_idx:
    Re_Dead.append(Re_Age[i])    # Re_Dead.append(titanic_df['Age'])

for j in Survive_idx:
    Re_Survived.append(Re_Age[j])# Re_Survived.append(titanic_df['Age'])

print Re_Dead
print Re_Survived

plt.figure(2, figsize = (15,5))


plt.subplot(1,4,1)

plt.hist(Re_Age,bins=8)
plt.title('People aboard vs Age')
plt.xlabel('Age')
plt.ylabel('People Aboard')
plt.ylim(0,500)


plt.subplot(1,4,2)
plt.hist(Re_Dead,bins=8)
plt.title('People Dead vs Age')
plt.xlabel('Age')
plt.ylabel('People Dead')
plt.ylim(0,500)


plt.subplot(1,4,3)
plt.hist(Re_Survived,bins=8)
plt.title('People Survived vs Age')
plt.xlabel('Age')
plt.ylabel('People Survived')
plt.ylim(0,500)

plt.subplot(1,4,4)

count1, division1 = np.histogram(Re_Dead, bins=[0,10,20,30,40,50,60,70,80])

# print count1

count2, division1 = np.histogram(Re_Survived, bins=[0,10,20,30,40,50,60,70,80])

# print count2

bar_hist = [0,10,20,30,40,50,60,70]

for idx in np.arange(len(bar_hist)):
    plt.text(bar_hist[idx], count1[idx]+5, '%i' % count1[idx])
    plt.text(bar_hist[idx], count2[idx]+count1[idx]+5, '%i' % count2[idx])
    # plt.text(count2[idx2], bar_hist[idx2], '%i')
    # plt.text(count1[idx]+0.5, yaxis_val[idx]+10, '{:.0%}' .format(yaxis_txt[idx]))


# print division1
# print count1
# print len(Re_Dead)

plt.title('Number of people who survived and drowned')

plt.axhline(y = Age_mean, linewidth = 2, color = 'green')

plt.bar([0,10,20,30,40,50,60,70], count1, width=10, color='b', label='Dead')
plt.bar([0,10,20,30,40,50,60,70], count2, width=10, bottom=count1, color='r', label='Survived')
plt.ylim(0,500)
plt.legend()
plt.legend(loc='upper left')

# Re_Survived = Series(Re_Survived)
# print Re_Survived.value_counts()

# hist, binh = plt.hist(Re_Dead, bins=8)
# plt.bar(hist, width=1)
# X = range(Re_Dead)
# plt.bar(X,Re_Dead, color='b')
# plt.bar(X,Re_Survived, color = 'r', bottom=Re_Dead)

##########################################################################


######################### Sex Dead, Survive rate #########################

plt.figure(3)
plt.title('Male and Female survive, Dead rate')
Surv = titanic_df['Survived'].tolist()

# print Surv
# print len(Surv)




male_idx = titanic_df[titanic_df['Sex'] == 'male'].index.tolist()
female_idx = titanic_df[titanic_df['Sex'] == 'female'].index.tolist()





male_d = []
male_s = []
female_d =[]
female_s = []

male_idx_d = []
female_idx_d = []


for idx in male_idx:
    if Surv[idx] == 0:
        male_d.append(Surv[idx])
        male_idx_d.append(idx)
    else:
        male_s.append(Surv[idx])

for idx in female_idx:
    if Surv[idx] == 0:
        female_d.append(Surv[idx])
        female_idx_d.append(idx)
    else:
        female_s.append(Surv[idx])

A_total = male_d + male_s + female_d + female_s

print 'male_d:%i, male_s:%i, female_d:%i, female_s:%i, Total : %i' % (len(male_d), len(male_s), len(female_d), len(female_s), len(male_d)+len(male_s)+len(female_d)+len(female_s))

n = 4
X = np.arange(n)
M_Total = len(male_idx)
F_Total = len(female_idx)
xaxis_name = ['Male Dead', 'Male Survived', 'Female Dead', 'Female Survived']
yaxis_val = [len(male_d), len(male_s), len(female_d), len(female_s)]
yaxis_txt = [float(len(male_d))/(M_Total+F_Total), len(male_s)/float(M_Total+F_Total), len(female_d)/float(F_Total+M_Total), len(female_s)/float(F_Total+M_Total)]

plt.bar(X[0:2], yaxis_val[0:2], width=1, color='r')
plt.bar(X[2:4], yaxis_val[2:4], width=1, color='b')

for idx in X:
    plt.text(X[idx]+0.5, yaxis_val[idx]+10, '{:.0%}' .format(yaxis_txt[idx]))

plt.xticks(X+0.5, xaxis_name)

# ave = (len(male_d)*100/(len(male_d)+len(male_s)))
# plt.bar(X[0], len(male_d), width=1)
# plt.text(X[0]+0.5, len(male_d) + 10, '{0}%'.format(ave))
# plt.bar(X[1], len(male_s), width=1)
# plt.bar(X[2], len(female_d), width=1)
# plt.bar(X[3], len(female_s), width=1)


# plt.bar([0], len(male_d), width=1)


################## Male, Female dead by Class ####################


plt.figure(4, figsize=(15,5))

# for idx in male_idx:
#     if titanic_df['Survived']
#

# m_dead = result[result['Survived'] == 0].index.tolist()
# f_dead = result_f[result_f['Survived'] == 0].index.tolist()


Pcls = titanic_df['Pclass'].tolist()


print len(male_idx)
print len(female_idx)

male_idx = male_idx_d
female_idx = female_idx_d

print len(male_idx)
print len(female_idx)


# print Pcls

male_class = []
female_class = []
male_cls_s = []
male_cls_d1 = []
male_cls_d2 = []
male_cls_d3 = []

female_cls_d1 = []
female_cls_d2 = []
female_cls_d3 = []

male_cls_d = []
female_cls_s = []
female_cls_d = []


for idx in male_idx:
    male_class.append(Pcls[idx])

for idx in female_idx:
    female_class.append(Pcls[idx])
#
# print male_class
# print female_class

for idx in male_class:
    # male_cls_d.append(Pcls[idx])
    # print Pcls[idx]

    if idx==1:
        male_cls_d1.append(Pcls[idx])
    elif idx==2:
        male_cls_d2.append(Pcls[idx])
    else:
        male_cls_d3.append(Pcls[idx])


for idx in female_class:

    if idx==1:
        female_cls_d1.append(Pcls[idx])
    elif idx==2:
        female_cls_d2.append(Pcls[idx])
    else:
        female_cls_d3.append(Pcls[idx])

# print len(male_class)
# print male_class.count()

male_dead_class1 = len(male_cls_d1)
male_dead_class2 = len(male_cls_d2)
male_dead_class3 = len(male_cls_d3)

female_dead_class1 = len(female_cls_d1)
female_dead_class2 = len(female_cls_d2)
female_dead_class3 = len(female_cls_d3)

# male portion passed away by Class
plt.subplot(1,2,1)
plt.title('Male dead boarding class')
plt.pie([male_dead_class1, male_dead_class2, male_dead_class3],labels = ['Class 1', 'Class 2', 'Class 3'], colors=['red', 'blue', 'Green'], autopct='%.0f%%')
plt.axis('equal')



# female portion passed away by Class
plt.subplot(1,2,2)
plt.title('Female dead boarding class')
plt.pie([female_dead_class1, female_dead_class2, female_dead_class3],labels = ['Class 1', 'Class 2', 'Class 3'], colors=['red', 'blue', 'Green'], autopct='%.0f%%')
plt.axis('equal')

##################################### Age Scatter ########################################





# make sure the value of resolution is a lowercase L,
#  for 'low', not a numeral 1


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api


# print male_idx
# print female_idx
#
# print len(male_idx)
# print len(female_idx)
plt.figure(5)
a = []
b = []
c = []
d = []


result = []
result_f =[]

for idx in male_idx:
    a.append(titanic_df.loc[idx])

for idx in female_idx:
    c.append(titanic_df.loc[idx])


result = pd.DataFrame(a)
result_f = pd.DataFrame(c)


result.index = np.arange(0,len(result))
result_f.index = np.arange(0,len(result_f))


result_male_idx = result[pd.notnull(result['Age'])].index.tolist()
result_female_idx = result_f[pd.notnull(result_f['Age'])].index.tolist()

# print result_male_idx
# print result_female_idx
# print result

###########################################################################

Age_mean = result['Age'].mean()
Age_mean = round(Age_mean,0)
print Age_mean

result['Age'] = result['Age'].fillna(Age_mean)

print len(result['Age'])


###########################################################################


for idx in result_male_idx:
    b.append(result.loc[idx])

for idx in result_female_idx:
    d.append(result_f.loc[idx])


male_age = pd.DataFrame(b)
female_age = pd.DataFrame(d)

# print male_age.head()
# print female_age.head()
#
# #
# #
# print len(b)

# print result.head()
# print result['Age'].loc[3]
# print pd.isnull(result['Age'].loc[3])

male_age_dist = plt.hist(x='Age', data=male_age, bins=80)
female_age_dist = plt.hist(x='Age', data=female_age, bins=80)


# print male_age_dist[0]


# print len(np.arange(80))
# print len(male_age_dist[0])
plt.figure(6, figsize=(15,5))

plt.subplot(1,2,1)
plt.title('Male dead age distribution')
# plt.axhline(y= )
plt.scatter(np.arange(80), male_age_dist[0], s=male_age_dist[0]*20, c=male_age_dist[0], cmap=plt.cm.Blues)
plt.xlim(0,80)
plt.ylim(0,20)

plt.subplot(1,2,2)
plt.title('Female dead age distribution')
plt.scatter(np.arange(80), female_age_dist[0], s=female_age_dist[0]*20, c=female_age_dist[0], cmap=plt.cm.YlOrRd)

plt.xlim(0,80)
plt.ylim(0,20)


############################################ Family number of members ##################################


# print result.head()
# print result_f.head()
plt.figure(7, figsize=(15,5))

family_m = []
family_f = []


for row in result.index:
    family_m.append(result['SibSp'][row] + result['Parch'][row])

for row in result_f.index:
    family_f.append(result_f['SibSp'][row]+result_f['Parch'][row])


# print result.head()

result['family'] = family_m
result_f['family'] = family_f

# print result.head()

plt.subplot(1, 2, 1)
plt.title("Number of family from male")
plt.ylim(0, 350)
sns.countplot(x='family', data=result)
histxt = np.histogram(result['family'], bins=np.arange(max(family_m)))

for idx, idy in zip(histxt[0], histxt[1]):
    plt.text(idy, idx, idx)


plt.subplot(1, 2, 2)
plt.title('Number of family from female')
plt.ylim(0, 350)
sns.countplot(x='family', data=result_f)
histxtf = np.histogram(result_f['family'], bins=np.arange(max(family_f)))

for idx, idy in zip(histxtf[0], histxtf[1]):
    plt.text(idy, idx, idx)



############################################# Embarked city with Map ######################################

plt.figure(8)
plt.title("Map by Embarked")

map = Basemap(llcrnrlat=35, llcrnrlon=-20, urcrnrlat=65, urcrnrlon=10, projection='cyl',
              lat_0=50, lon_0=-5, resolution='i')

# print result.head()
# print result_f.head()

mcherbourg = 0.0
mqueenstown = 0.0
msouthampton = 0.0

mcherbourg_r = 0.0
mqueenstown_r = 0.0
msouthampton_r = 0.0

fcherbourg = 0.0
fqueenstown = 0.0
fsouthampton = 0.0

fcherbourg_r = 0.0
fqueenstown_r = 0.0
fsouthampton_r = 0.0


map.drawcoastlines()
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral', lake_color='aqua')

x, y = map(0, 0)

m_dead=[]
f_dead=[]


male_dead = []
female_dead = []

m_dead = result[result['Survived'] == 0].index.tolist()
f_dead = result_f[result_f['Survived'] == 0].index.tolist()

# print len(m_dead)
# print len(f_dead)



for idy in m_dead:
    male_dead.append(result.loc[idy])

male_dead_frame = pd.DataFrame(male_dead)

for idy in f_dead:
    female_dead.append(result_f.loc[idy])

female_dead_frame = pd.DataFrame(female_dead)

# print male_dead_frame.head()
# print female_dead_frame.head()

for idx in male_dead_frame['Embarked']:

    if idx == 'C':
        mcherbourg += 1

    elif idx == 'Q':
        mqueenstown += 1

    elif idx == 'S':
        msouthampton += 1



for idx in female_dead_frame['Embarked']:

    if idx == 'C':
        fcherbourg += 1

    elif idx == 'Q':
        fqueenstown += 1

    elif idx == 'S':
        fsouthampton += 1




mtotal = mcherbourg + mqueenstown + msouthampton
ftotal = fcherbourg + fqueenstown + fsouthampton

print "male dead total: %f, cherbourg: %f, queenstown: %f, southampton: %f" % (mtotal, mcherbourg, mqueenstown, msouthampton)

print "female dead total: %f, cherbourg:  %f, queenstown: %f, southampton: %f " % (ftotal, fcherbourg, fqueenstown, fsouthampton)




mcherbourg_r = mcherbourg / float(mtotal)
mqueenstown_r = mqueenstown / float(mtotal)
msouthampton_r = msouthampton / float(mtotal)


fcherbourg_r = fcherbourg / float(ftotal)
fqueenstown_r = fqueenstown / float(ftotal)
fsouthampton_r = fsouthampton / float(ftotal)


print "dead male rate from Cherbourg : %f, from Queenstown : %f, from Southampton : %f " % (mcherbourg_r, mqueenstown_r, msouthampton_r)

print "dead female rate from Cherbourg : %f, from Queenstown : %f, from Southampton : %f " % (fcherbourg_r, fqueenstown_r, fsouthampton_r)


xloc = [-6, -1, -1]
yloc = [53, 49, 51]
color = ['bo', 'ro', 'yo']
labels = ['Queenstown','Cherbourg','Southampton']


x, y = map(xloc,yloc)
for i in np.arange(len(color)):
    map.plot(xloc[i],yloc[i], color[i], markersize=10)


for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt, ypt, label)

plt.figure(9, figsize=(15,5))

plt.subplot(1, 2, 1)
plt.title("Dead Male Pie Chart by city")
plt.pie([mcherbourg_r, mqueenstown_r, msouthampton_r], labels= ['Cherbourg', 'Queenstown', 'Southampton'] , colors=['red','blue','yellow'], autopct='%1.1f%%')
plt.axis('equal')


plt.subplot(1, 2, 2)
plt.title("Dead Female Pie Chart by city")
plt.pie([fcherbourg_r, fqueenstown_r, fsouthampton_r], labels= ['Cherbourg', 'Queenstown', 'Southampton'] , colors=['red','blue','yellow'], autopct='%1.1f%%')
plt.axis('equal')

# plt.colorbar()

plt.show()
#
```
