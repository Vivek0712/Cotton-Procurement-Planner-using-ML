#OE COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.OE_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.OE_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.OE_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.OE_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'OE':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')

writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='OE', index=False,
             startcol=1,startrow=1)

writer.save()


#ME COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.ME_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.ME_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.ME_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.ME_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'ME':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='ME', index=False,
             startcol=1,startrow=1)

writer.save()


#MC COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.MC_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.MC_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.MC_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.MC_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'MC':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='MC', index=False,
             startcol=1,startrow=1)

writer.save()

#DC COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.DC_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.DC_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.DC_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.DC_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'DC':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='DC', index=False,
             startcol=1,startrow=1)

writer.save()

#IM-1 COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.IM1_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.IM2_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.IM1_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.IM1_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))
#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'IM1':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='IM1', index=False,
             startcol=1,startrow=1)

writer.save()
#IM2 COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.IM2_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.IM2_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.IM2_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.IM2_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'IM2':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='IM2', index=False,
             startcol=1,startrow=1)

writer.save()

#IM3 COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.IM3_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.IM3_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.IM3_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.IM3_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'IM3':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='IM3', index=False,
             startcol=1,startrow=1)

writer.save()

#IM4 COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.IM4_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.IM4_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.IM4_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.IM4_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'IM4':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='IM4', index=False,
             startcol=1,startrow=1)

writer.save()

#IM5 COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.IM5_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.IM5_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.IM5_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.IM5_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'IM5':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='IM5', index=False,
             startcol=1,startrow=1)

writer.save()

#IM6 COTTON PREDICTION
import pandas as pd
import numpy as np
location = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'
df = pd.read_csv(location)
X =[df.IM6_ACT.values][0]
X= np.asarray(X)
X=X.reshape(len(X),-1)
y= df.IM6_ACT.values
dates = df.YEAR.values


import numpy as np

X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y[np.isnan(y)] = np.median(y[~np.isnan(y)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(X, y)
res = nn.predict(X)
#three year prediction
predX =[res][0]
predX= np.asarray(predX)
predX=predX.reshape(len(predX),-1)
predy= res

#dates = df.YEAR.values


import numpy as np

predX[np.isnan(predX)] = np.median(X[~np.isnan(predX)])
predy[np.isnan(predy)] = np.median(y[~np.isnan(predy)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(predX, predy)
#print(nn.predict(predX))

#inclusion of marketing team prediction

marX =[df.IM6_MAR.values][0]
marX= np.asarray(marX)
marX=marX.reshape(len(marX),-1)
mary= df.IM6_MAR.values
dates = df.YEAR.values


import numpy as np

marX[np.isnan(marX)] = np.median(marX[~np.isnan(marX)])
mary[np.isnan(mary)] = np.median(mary[~np.isnan(mary)])
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', 
solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, 
power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08)
n = nn.fit(marX, mary)
print(nn.predict(marX))

#writing prediction values to excel

fn = r'C:\Users\Harish\Desktop\cotton_procurement\Cotton_year1.csv'

df = pd.read_excel(fn, header=None)
#marX= np.asarray(marX)
#marX=marX.reshape(len(marX),-1)
df2 = pd.DataFrame({'IM6':nn.predict( marX)})
df2= df2.transpose()

writer = pd.ExcelWriter(fn, engine='openpyxl')
#
#
writer.sheets = dict((ws.title, ws) for ws in fn)

df.to_excel(writer, sheet_name='prediction', index=False)
df2.to_excel(writer, sheet_name='prediction', header='IM6', index=False,
             startcol=1,startrow=1)

writer.save()