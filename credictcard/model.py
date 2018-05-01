from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline
import seaborn as sns
from numpy import log
from sklearn.linear_model import LogisticRegression  
import math

data = pd.read_csv('cs_training.csv')
data.columns = ['Unnamed: 0', 'Y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']

data = data[data.x2>20]
data = data[data.x2<91]
data = data[data.x3<50]
data = data[data.x7<50]


index1 = data[data['x5'].isnull()].index
ageList = data['x2'].unique()
fillDict = {}
for age in ageList:
    fillDict[age] = data[data.x2==age]['x5'].median()

def fill_monthIncome(data,index1):
    for i in index1:
        age = data.loc[i,'x2']
        fill_value = fillDict[age]
        data.loc[i,'x5'] = fill_value
fill_monthIncome(data.index1)
data.to_csv('clean_data.csv')



data = pd.read_csv('clean_data.csv')
names = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
X = data.loc[:,['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
Y = data['Y']

list1 = list(data.x2.unique())
list1 = sorted(list1)
IncomeMedian = []
for i in list1:
    result = data[data.x2==i]['x5'].median()
    IncomeMedian.append(result)
 
def get_value(data,feature):
    test1 = list(data[feature].unique())
    test1 = sorted(test1)
    return test1

total_good = len(data)-data['Y'].sum()
total_bad = data['Y'].sum()    
total_ratio = data['Y'].sum()/(len(data)-data['Y'].sum())
def compute_woe1(data,feature,n):
    woe_dict ={}
    iv = 0
    total_list = data[feature].value_counts()
    index1 = get_value(data,feature)
    for i in index1:
        if i <= n:
            bad = data[data[feature]==i]['Y'].sum()
            good = total_list[i] - bad
            result = bad/good
            woe = log(result/total_ratio)
            woe_dict[i] = woe
            iv_test = (bad/total_bad - good/total_good)*woe
            iv = iv+iv_test
        else:
            bad = data[data[feature]>=i]['Y'].sum()
            good = len(data[data[feature]>=i]['Y']) - bad
            result = bad/good
            woe = log(result/total_ratio)
            woe_dict[i] = woe
            iv_test = (bad/total_bad - good/total_good)*woe
            iv = iv+iv_test
            break
    return woe_dict,iv
 


data.x1 = pd.qcut(data.x1,10,labels=[0,1,2,3,4,5,6,7,8,9])
data.x2 = pd.cut(data.x2,bins=[20,25,30,35,40,50,60,70,90],labels=[0,1,2,3,4,5,6,7])
data.x4 = pd.qcut(data.x4,20,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
data.x5 = pd.qcut(data.x5,20,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

woe_x1,IV1 = compute_woe1(data,'x1',100)
woe_x2,IV2 = compute_woe1(data,'x2',100)
woe_x3,IV3 = compute_woe1(data,'x3',6)
woe_x4,IV4 = compute_woe1(data,'x4',100)
woe_x5,IV5 = compute_woe1(data,'x5',100)
woe_x6,IV6 = compute_woe1(data,'x6',20)
woe_x7,IV7 = compute_woe1(data,'x7',5)
woe_x8,IV8 = compute_woe1(data,'x8',7)
woe_x9,IV9 = compute_woe1(data,'x9',4)
woe_x10,IV10 = compute_woe1(data,'x10',5)

index1 = data[data.x3>=4].index
data.loc[index1,'x3'] = 4
index2 = data[data.x7>=4].index
data.loc[index2,'x7'] = 4
index3 = data[data.x9>=4].index
data.loc[index3,'x9'] = 4

IVList = [IV1, IV2, IV3, IV4, IV5, IV6, IV7, IV8, IV9, IV10]
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1
ax1.bar(x, IVList, width=0.4)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV(Information Value)', fontsize=14)
for a, b in zip(x, IVList):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
plt.show()


def convert_woe(feature,woe):
    list1 = []
    for i in data[feature]:
        if i in woe.keys():
            list1.append(woe[i])
        else:
            list1.append(woe[(len(woe)-1)])
    return list1
    
data.x1 = convert_woe('x1',woe_x1)
data.x2 = convert_woe('x2', woe_x2)
data.x3 = convert_woe('x3', woe_x3)
data.x7 = convert_woe('x7', woe_x7)
data.x9 = convert_woe('x9', woe_x9)
data = data.drop(['x4','x5','x6','x8','x10'],axis=1)
data = data.loc[:,['Y','x1','x2','x3','x7','x9']]
Y = data.Y
X = data.loc[:,['x1','x2','x3','x7','x9']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)




lr = LogisticRegression(C=0.1)
lr.fit(X_train,Y_train)
#lr.coef_
y_pred = lr.predict(X_test)
fpr, tpr, thresholds = roc_curve(Y_test,y_pred)  
roc_auc = auc(fpr,tpr)  
# Plot ROC  
plt.title('Receiver Operating Characteristic')  
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)  
plt.legend(loc='lower right')  
plt.plot([0,1],[0,1],'r--')  
plt.xlim([-0.1,1.0])  
plt.ylim([-0.1,1.01])  
plt.ylabel('True Positive Rate')  
plt.xlabel('False Positive Rate')  
plt.show()  

coe=[9.738849,0.638002,0.505995,1.032246,1.790041,1.131956]
p = 20 / math.log(2)
q = 600 - 20 * math.log(20) / math.log(2)
baseScore = round(q + p * coe[0], 0)

data.x1 = convert_woe('x1',woe_x1)
data.x2 = convert_woe('x2', woe_x2)
data.x3 = convert_woe('x3', woe_x3)
data.x7 = convert_woe('x7', woe_x7)
data.x9 = convert_woe('x9', woe_x9)

data.x1 = round(data.x1*p*coe[1])
data.x2 = round(data.x2*p*coe[2])
data.x3 = round(data.x3*p*coe[3])
data.x7 = round(data.x7*p*coe[7])