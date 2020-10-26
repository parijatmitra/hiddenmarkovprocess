# Fetch the data from Firebase
import numpy as np
from firebase import firebase
import math
firebase = firebase.FirebaseApplication('https://androidbletutorial.firebaseio.com/',None)
All_Sample = []
result = firebase.get('',None)
length = 10

data_rx=np.array(0)
#print(result)

# Define Partition Function
def Partition(value):
    end_value = pow(2,16)-1  # maximam value
    global length
    num_level = length
    return_value = math.ceil(value*num_level/end_value) - 1
    #print(value,"  ", return_value);
    return return_value
    
    
sequence = dict()
vedba = dict()
addresses = sequence.keys()

# Sample value
for key in result.keys():
  mac_address = result[key]["text"].split(":")[1][0:14]
  #print("The MAC Address is",mac_address)
  temp = result[key]["text"].split(":")[2]
  x1, x2 = temp.split(" ")[1], temp.split(" ")[2]
  y1, y2 = temp.split(" ")[3], temp.split(" ")[4]
  z1, z2 = temp.split(" ")[5], temp.split(" ")[6]

  x = (int(x1)<<8) + int(x2)
  y = (int(y1)<<8) + int(y2)
  z = (int(z1)<<8) + int(z2)

  if mac_address not in addresses:
        sequence[mac_address]=[]
        addresses=sequence.keys()
        vedba[mac_address]=[]

  sample = math.sqrt(x*x+y*y+z*z)
  sample = Partition(int(sample))
  vedba[mac_address].append(sample)


#print(sequence)
#print(vedba)

## determining the n component
from hmmlearn import hmm
def n_components_cow(X):
    X = np.array(X).reshape(-1,1)
    accu_racy=[]
    for i in range(2,5):
        remodel = hmm.MultinomialHMM(n_components=i, n_iter=1000)
        remodel.fit(X)
        yi= remodel.predict(X)
        
        # importing necessary libraries 
        from sklearn import datasets 
        from sklearn.metrics import confusion_matrix 
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB 
        
        X_train, X_test, yi_train, yi_test = train_test_split(X, yi, random_state = 0) 
        gnb = MultinomialNB().fit(X_train, yi_train) 
        gnb_predictions = gnb.predict(X_test)      
        accuracy= gnb.score(X_test, yi_test) 
        accu_racy.append(accuracy)
     
    #print(accu_racy,end='')
    #print('\n')
    #print( max(accu_racy))
    min_ele,max_ele=accu_racy[0],accu_racy[0] 
    
    for i in range(1,len(accu_racy)): 
        if accu_racy[i]>max_ele: 
            max_ele=accu_racy[i] 
   # print('no of componenets list',accu_racy,'is',max_ele)  
    n_components= accu_racy.index(max(accu_racy)) 
    return n_components+2
    



# find transition matrix and emission matrix
#defining states and sequence symbols
import numpy as np
import decimal
from decimal import Decimal
import math
import time
from hmmlearn.hmm import MultinomialHMM

transition_probability = np.array([[0.7,0.3],[0.1,0.9]])
emission_probability = np.array([[0.02, 0.16, 0.06, 0.12, 0.1, 0.1, 0.08, 0.14, 0.04, 0.18],
                                 [0.14, 0.04, 0.18, 0.02, 0.16, 0.06, 0.12, 0.1, 0.1, 0.08]])
                                 
i = 0                                
X = vedba[list(sequence.keys())[i]]
n_components = int(n_components_cow(X))


remodel = MultinomialHMM(n_components=n_components, tol=1e-10, algorithm='viterbi',n_iter=5000, init_params="s")
remodel.transmat_ = transition_probability
remodel.emissionprob_ = emission_probability
    
    
for key in sequence.keys():
    print("mac_address: ",key," n_components: ",n_components)
    temp = [vedba[key]]
    remodel.fit(temp)
    
    print(remodel.monitor_)
    print(remodel.transmat_)
    print(remodel.emissionprob_)
    print("\n")
    # new trainsition and emission matrix
    transition_probability = (remodel.transmat_).tolist()
    emission_probability = (remodel.emissionprob_).tolist()
    i += 1
    X, Z = remodel.sample(100)
    print(X,Z)
