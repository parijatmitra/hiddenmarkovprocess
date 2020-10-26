def n_components_cow():
    
    BIC=[]
    for i in range(2,10):
        from hmmlearn import hmm
        remodel = hmm.MultinomialHMM(n_components=i, n_iter=1000)
        remodel.fit(X)
        yi= remodel.predict(X)
        
        
        from sklearn import datasets 
        from sklearn.metrics import confusion_matrix 
        from sklearn.model_selection import train_test_split 
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from math import log
        
        X_train, X_test, yi_train, yi_test = train_test_split(X, yi, random_state = 0)
        
        model = LinearRegression()
        model.fit(X_train, yi_train)
        num_params = len(model.coef_) + 1
        
        
        
        yhati = model.predict(X_test)
        mse = mean_squared_error(yi_test, yhati)
        bic = len(yi)* log(mse) + num_params * log(len(yi))
        
        print('BIC score for n_components ={}'.format(i))
        
        print(bic)
        BIC.append(bic)
    print("\nlist of BIC SCORE IS ")
    print(BIC,end='')
    print('\n')
    print("minimum value in list of BIC scores")
    print( min(BIC))
    min_ele,max_ele=BIC[0],BIC[0] 
        
    for i in range(1,len(BIC)):
        if BIC[i]<min_ele: 
            min_ele=BIC[i]
    
    min_ele=BIC[i]
    n_components= BIC.index(min(BIC)) 
    print('\n number of components is = ')
    return n_components+2
n_components_cow() 
            
            
        
       
        
        
      