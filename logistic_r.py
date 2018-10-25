import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:
    def __init__(self,normalize=True,add_bais=True,learning_rate=0.1,toll=0.0001,max_itr=100):
        self.normalize = normalize
        self.add_bais = add_bais
        self.learning_rate = learning_rate
        self.toll = toll
        self.max_itr = max_itr
        self.th = []
        self._min = 0
        self._max = 0
        self._nclass = 0
        
    def fit(self,X,y):
        self._min = X.min()
        self._max = X.max()
        m = y.shape[0]
        
        if(self.normalize):
            X = (X - self._min) / (self._max - self._min)
        
        if(self.add_bais):
            X = np.c_[np.ones(m), X]
            
        th = np.zeros((X.shape[1],1))
        
        self._nclass = np.unique(y).shape[0]
        
        if(self._nclass == 2):
            i = 0
            while (i < self.max_itr):
                th = th - (self.learning_rate / m) * (X.T).dot(1/(1+np.exp(-X.dot(th))) - y)
                i+=1
            self.th = th
            
        elif(self._nclass > 2):
            for i in range(self._nclass):
                y_m = np.where(y_train == i, 1, 0)
                i = 0
                while (i < self.max_itr):
                    th = th - (self.learning_rate / m) * (X.T).dot(1/(1+np.exp(-X.dot(th))) - y_m)
                    i+=1
                self.th.append(th)
        return self.th
    
    
    def predict(self,X):
        if(self.normalize):
            X = (X - self._min) / (self._max - self._min)
            
        if(self._nclass == 2):
            if(self.normalize):
                X = (X - self._min) / (self._max - self._min)
            if(self.add_bais):
                return np.where((1/(1+np.e**(-np.c_[np.ones(X.shape[0]), X].dot(self.th)))) > 0.5, 1, 0)
            else:
                return np.where(X.dot(self.th) > 0.5, 1, 0)
            
        else:
            nptl = []
            for i in range(self._nclass):
                thv = self.th[i]
                if(self.add_bais):
                    nptl.extend(np.where((1/(1+np.exp(-np.c_[np.ones(X.shape[0]), X].dot(thv)))) > 0.5, i, 0).T.tolist())
                else:
                    nptl.extend(np.where(X.dot(thv) > 0.5, i, 0).T.tolist())
            nptl = [max(x) for x in list(zip(*nptl))]
            return np.array(nptl).reshape(-1,1)


dt = np.loadtxt('data/data.csv',delimiter=',')
dt_x = dt[:,:2]
dt = np.c_[dt_x,dt_x[:,0]*dt_x[:,0],dt[:,2:]]
np.random.shuffle(dt)
n = dt.shape[0]
x_train = dt[:int(n*.8),:3]
y_train = dt[:int(n*.8),3:]
x_test = dt[int(n*.8):,:3]
y_test = dt[int(n*.8):,3:]


lr = LogisticRegression(max_itr=1000,toll=-1,learning_rate=0.5,normalize=False)
lr.fit(x_train,y_train)
pv = lr.predict(x_test)
print(np.c_[y_test,pv])

misc = sum(np.where(y_test - pv.reshape(-1,1) != 0,1,0))
acc = 1 - misc / (n-int(n*.8))
print('accuracy for test data : {}'.format(acc))
