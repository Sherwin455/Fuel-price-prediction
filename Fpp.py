//Step1: Prepare the dataset
import pandas as pd df=pd.read_csv("C:\\fuel_data.csv") df

//Step2:Import Dataset Using Pandas
df.head()
Driven KM	fuel Amount	
0		390.0	3600.0
1		403.0	3705.0
2		396.5	3471.0
3		383.5	3250.5
4		321.1	3263.7
df.shape	
(19, 2)
df.info()	
<class 'pandas.core.frame.DataFrame'>	
RangeIndex: 19 entries, 0 to 18	
Data columns (total 2 columns):	
#	Column	Non-Null Count Dtype	
0	drivenKM	19 non-null	float64
1	fuelAmount 19 non-null	float64 dtypes: float64(2)
memory usage: 432.0 bytes

//Step 3. [Preprocessing] Checking Value For Null
df.isnull()


Driven KM	Fuel Amount
0	False	False
1	False	False
 
Driven KM	fuel Amount


2	False	False
3	False	False
4	False	False
5	False	False
6	False	False
7	False	False
8	False	False
9	False	False
10	False	False
11	False	False
12	False	False
13	False	False
14	False	False
15	False	False
16	False	False
17	False	False
18	False	False


//Step4.[Visualize the Relationship]

import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
import numpy as np sns.relplot(x="drivenKM",y="fuelAmount",data=df,kind='scatter')

<seaborn.axisgrid.FacetGrid at 0x1f0166dc7f0>
 

//Step5.[Prepare X matrix and y vector]. Extract "drivenKM" coloumn and store into dataframe X. Extract "fuelAmount" and store into y]

y=df.fuelAmount data1=['drivenKM'] X=df[data1]

//Step6.[Examine X and y]. Print X,y type of X and y.]

print(X)
driven KM
0	390.00
1	403.00
2	396.50
3	383.50
4	321.10
5	391.30
6	386.10
7	371.80
8	404.30
9	392.20
10	386.43
11	395.20
12	381.00
 
14	397.00	
15	407.00	
16	372.40	
17	375.60	
18	399.00	
print(y)
0	3600.0	
1	3705.0	
2	3471.0	
3	3250.5	
4	3263.7	
5	3445.2	
6	3679.0	
7	3744.5	
8	3809.0	
9	3905.0	
10	3874.0	
11	3910.0	
12	4020.7	
13	3622.0	
14	3450.5	
15	4179.0	
16	3454.2	
17	3883.8	
18
Name:	4235.9
fuelAmount,	
dtype: float64

//Step7. [Split Dataset] Split the dataset into train and test split()
print(y.dtypes) float64
print(X.dtypes)
drivenKM	float64 dtype: object

print(type(y))
<class 'pandas.core.series.Series'>
print(type(X))
<class 'pandas.core.frame.DataFrame'>
from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2)
X_train, X_test, y_train, y_test

(	drivenKM
12	381.00
15	407.00
7	371.80
17	375.60
8	404.30
4	321.10
2	396.50
14	397.00
1	403.00
 
10	386.43
5	391.30
9	392.20
11	395.20
3	383.50,
	drivenKM
6	386.1
16	372.4
18	399.0
0	390.0,
12	4020.7
15	4179.0
7	3744.5
17	3883.8
8	3809.0
4	3263.7
2	3471.0
14	3450.5
1	3705.0
13	3622.0
10	3874.0
5	3445.2
9	3905.0
11	3910.0
3	3250.5
Name: fuelAmount, dtype: float64, 6	3679.0
16	3454.2
18	4235.9
0	3600.0
Name: fuelAmount, dtype: float64)

 
X_train.shape
(15, 1)
X_test.shape
(4, 1)

y_train.shape
(15,)
y_test.shape
(4,)

Part-I. Linear Regression Baseline Model
 
//Step8. [Build Model]. Create Linear Regression model and train with fit() using X_train and y_train values.

from sklearn.linear_model import LinearRegression model = LinearRegression() model.fit(X_train,y_train)
LinearRegression()

Step9.[Predict price for 800 km] If I need to travel 800 km how much do I
 
need to Spend on fuel
new = [[800]]
y_n= model.predict(new) y_n

array([6749.03598132])


//Step13. [Build LR model]. Create a new LR model, fit on scaled X_train and predict on scaled X_test

model1 = LinearRegression() model1.fit(ss,y_train) LinearRegression()

s_y_pred = model1.predict(ss1) s_y_pred
array([3814.29479572, 3642.76915571, 3805.04237531, 3869.80931822])

Step14. [Print Mean Squared Error and R2 Error]. What is the output?. MSE reduced or not?. Why?.

mean_squared_error(y_test,s_y_pred)
42030.87985447279
r2_score(y_test,s_y_pred)
-0.032882281579435624
Step15. [Plot scatter plot]. Display Scatter Plot between actual y (aka ground truth) vs predicted y values. That is, between y_test and y_pred.

plt.scatter(y_test,y_pred)
<matplotlib.collections.PathCollection at 0x25aa5d30130>




 
Part-III. Linear Regression with Scaling using MinMaxScaler and Comparison with KNeighborsRegressor and SGDRegressor


Step16. [Repeat with MinmaxScaler]. Repeat scaling using MinMaxScaler, LR model creation, fit, predict and error computation steps.

from sklearn.preprocessing import MinMaxScaler m_scaler = MinMaxScaler()
m_ss = m_scaler.fit_transform(X_train) m_ss

array([[0.95343423],
[0.88358556],
[0.90686845],
[0.59254948],
[0.76053551],
[0.75669383],
[0.63445867],
[0.80209546],
[1.	],
[0.82770664],
[0.72642608],
[0.69732247],
[0.59022119],
[0.81722934],
[0.	]])

m_ss1 = m_scaler.transform(X_test) m_ss1

array([[0.87776484],
[0.59720605],
[0.86263097],
[0.9685681 ]])

model2 = LinearRegression() model2.fit(m_ss,y_train)

LinearRegression()

ms_y_pred = model2.predict(m_ss1) ms_y_pred

array([3814.29479572, 3642.76915571, 3805.04237531, 3869.80931822])

mean_squared_error(y_test,ms_y_pred)

42030.87985447284

r2_score(y_test,ms_y_pred)

-0.032882281579436734
 
Step17. [Compare KNN Regressor]. Repeat the above steps for KNeighborsRegressor model and compare MSE of LR with KNN Regressor.

from sklearn.neighbors import KNeighborsRegressor

mod_neigh = KNeighborsRegressor(n_neighbors=5) mod_neigh.fit(X, y)	
KNeighborsRegressor()	
n_y_pred = mod_neigh.predict(X)	
n_y_pred	
array([3700.64, 3875.88, 3794.48, 3684.84,	3593.64, 3746.84, 3684.84,
3745.04, 3875.88, 3666.24, 3569.74,	3794.48, 3741.6 , 3745.04,
3794.48, 3875.88, 3745.04, 3745.04,	3754.48])

knn_mse=mean_squared_error(y,n_y_pred) knn_mse

70460.30507368421

r2_score(y,n_y_pred)

0.06403925984775638

Step18. [Compare SGD Regressor]. Repeat the above steps for SGDRegressormodel and compare MSE of LR with SGD Regressor.

from sklearn.linear_model import SGDRegressor from sklearn.pipeline import make_pipeline

reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3)) reg.fit(X, y)

Pipeline(steps=[('standardscaler', StandardScaler()),
('sgdregressor', SGDRegressor())])

r_y_pred = reg.predict(X) r_y_pred

array([3740.57047646, 3830.21877598, 3785.39462622, 3695.7463267 ,
3265.43448899, 3749.53530641, 3713.6759866 , 3615.06285713,
3839.18360593, 3755.74172714, 3715.9516742 , 3776.42979627,
3678.50626909, 3616.44206173, 3788.84263774, 3857.80286814,
3619.20047095, 3641.26774468, 3802.63468382])

sgdr_mse=mean_squared_error(y,r_y_pred)sgdr_mse 58823.8671411817
 
r2_score(y,r_y_pred) 0.21861209413581573
Step19. [Select best model]. Tabulate MSE values of LR, KNNR and SGDR andselect the model with the lowest MSE.

IN[ ]:
print("LR model ",lr_mse) print("KNNR model ",knn_mse) print("SGDR model ",sgdr_mse)


OUT[ ]:
LR model 42030.879854472776
KNNR model 70460.30507368421
SGDR model 58823.8671411817
