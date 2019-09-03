# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1
loans_data = pd.read_excel("Loans_Data_New.xlsx")

loans_data.describe()

def status(x) : 
    return pd.DataFrame([x.median(),x.mad(),x.var(),x.std(),x.skew(),x.kurt()],
                        index=['median','mad','var','std','skew','kurt'])
    
status(loans_data)

dummy = pd.get_dummies(loans_data,drop_first = True)
    
corr = dummy.corr()
    
print(corr['debt_settlement_flag_Y'].sort_values(ascending=False))


#2
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def loss_logsit(x_test, t ,beta):
    X_test_add_one = np.column_stack((x_test,np.ones(len(x_test))))
    
    model_0 = sigmoid(np.dot(X_test_add_one, beta))
    
    loss_temp = t * np.log(model_0) + (1-t) * np.log(1-model_0) 
    
    return  -( np.sum(loss_temp) / len(x_test) )

def myLogisticGD(X, t, beta, alpha, numIterations):
    
    X_add_one = np.column_stack((X,np.ones(len(X))))
    
    for i in range(0,numIterations):
        hypothesis = np.dot(X_add_one,beta)
        
        h = sigmoid(hypothesis)

        loss = h - t 
        
       
        cost = loss_logsit( X, t,  beta)
        
        loss_total[i]=cost
#        print("Iteration %d | Cost: %f" % (i, cost))
        
        gradient = np.dot(X_add_one.transpose(), loss) / len(X)
        
        beta = beta - alpha * gradient
        
        beta_total[i,:]= beta.transpose()
        
    return beta

def show_loss_and_beta(loss_total,beta_total):
    
    fig1 = plt.figure()
    plt.plot(loss_total, label = "Loss fucntion")
    plt.plot(beta_total[:,0], label = "Beta0")
    plt.plot(beta_total[:,1], label = "Beta1")
    plt.plot(beta_total[:,2], label = "Beta2")
    plt.legend(loc="upper right")
    plt.xlabel("Number of iteration")
    plt.show()


from sklearn.preprocessing import StandardScaler

#标准化，返回值为标准化后的数据


#c = preprocessing.scale( loans_data['annual_inc'].values.reshape(len(loans_data['annual_inc']),1) )
   
c=  pd.Series(preprocessing.scale( loans_data['annual_inc']) )
#X_loans = loans_data[['annual_inc','int_rate']]
X_loans = pd.DataFrame((c,loans_data['int_rate'])).transpose()
t_loans = pd.get_dummies(loans_data['debt_settlement_flag'],drop_first=True).iloc[:,0]

init_beta = [0.1,0.1,0.1]

alpha = 0.0005

numIterations = 10000

loss_total= np.zeros((numIterations,1))
beta_total= np.zeros((numIterations,3))


#X_add_one = np.column_stack((X_loans,np.ones(len(X_loans))))
#
#hypothesis = np.dot(X_add_one,init_beta)
#
#h = sigmoid(hypothesis)
#xTrans = X_add_one.transpose()
#
#
#for i in range(0,numIterations):
#    hypothesis = np.dot(X_add_one,init_beta)
#
#    h = 1.0/(1+np.exp(-hypothesis))
#
#    loss = t_loans - h
#
#    gradient = np.dot(xTrans, loss) / len(X_loans)
#
#    beta = init_beta - alpha * gradient

myLogisticGD(X_loans,t_loans,init_beta,alpha,numIterations)

show_loss_and_beta(loss_total,beta_total)

#alphas = np.linspace(0.1,1,10)
#alphas = np.insert(alphas,0,[0.001,0.005,0.0001,0.0005])
#alphas = np.append(alphas,[5,10,50,100,500,1000])
#alphas = 0.1




#def loss(x_test, t ,beta):
#    
#    X_test_add_one = np.column_stack((x_test,np.ones(len(x_test))))
#    h = sigmoid(np.dot(X_test_add_one, beta))
#    return (-t * np.log(h) - (1 - t) * np.log(1 - h)).mean()




#for i in alphas:
#    print("alpha: " ,i)
#    print("beta",myLogisticGD(X_loans,t_loans,init_beta,i,50))


#X_add_one = np.column_stack((X_loans,np.ones(len(X_loans))))
#h = sigmoid(np.dot(X_add_one, init_beta))
#t = t_loans
#def loss(x_test, t ,beta):
#    
#    X_test_add_one = np.column_stack((x_test,np.ones(len(x_test))))
#    h = sigmoid(np.dot(X_test_add_one, beta))
#    return (-t * np.log(h) - (1 - t) * np.log(1 - h)).mean()



    
# 3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_sklearn =  dummy[['loan_status_Fully Paid','term_ 60 months','total_rec_int']]
y_sklearn = dummy['debt_settlement_flag_Y']

X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(X_sklearn,y_sklearn,test_size=0.2,random_state=0)
    
clf = LogisticRegression(random_state=0,fit_intercept = True )

clf.fit(X_sklearn,y_sklearn)

y_log_estimate = clf.predict(X_log_test)



#def loss_logsit(x_test, t ,beta):
#    X_test_add_one = np.column_stack((x_test,np.ones(len(x_test))))
#    
#    model_0 = sigmoid(np.dot(X_test_add_one, beta))
#    
#    loss_temp = t * np.log(model_0) + (1-t) * np.log(1-model_0) 
#    
#    return  -( np.sum(loss_temp) / len(x_test) )


beta = np.append(clf.coef_,clf.intercept_)

loss_logsit(X_log_test,y_log_test,beta)




