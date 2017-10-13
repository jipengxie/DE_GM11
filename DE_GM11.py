#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def DE_GM11(x0):
    import numpy as np
    Gm = 200
    F0 = 0.5
    Np = 500
    CR = 0.9
    G = 0
    D = 2
    
    CRmin = 0.1
    CRmax = 0.6
    Fmin = 0.2
    Fmax = 0.9

    Gmin = np.zeros([1,Gm])
    best_x = np.zeros([Gm,D])
    value = np.zeros([1,Np])
    xmin = np.array([-1,20])
    xmax = np.array([1,345])
    
    x1 = x0.cumsum() 
    z1 = (x1[:len(x1)-1] + x1[1:])/2.0 
    z1 = z1.reshape((len(z1),1))
    B = np.append(-z1, np.ones_like(z1), axis = 1)
    Yn = x0[1:].reshape((len(x0)-1, 1))

    f=lambda v: sum((Yn-(np.dot(B,v)).reshape(len(np.dot(B,v)),1))**2)
    X0 = (xmax-xmin)*np.random.uniform(size=(Np,D)) + xmin
    XG = X0

    XG_next_1 = np.zeros([Np,D])
    XG_next_2 = np.zeros([Np,D]) 
    XG_next = np.zeros([Np,D])
    trace = np.zeros([Gm,2])


    while G<Gm:
        for i in range(Np):
            a=1
            b=Np
            A=np.array(range(b-a+1))
            np.random.shuffle(A)
            dx = A + a-1
            j = dx[0]
            k = dx[1]
            p = dx[2]
            if j == i:
                j = dx[3]
            elif k == i:
                k = dx[3]
            elif p == i:
                p = dx[3]
            suanzi = np.exp(1-Gm/(Gm + 1-G))  
            F = F0*2**suanzi
    ##        F = F0
    ##        F = Fmin+(Fmax-Fmin)*Fsuanzi

            son = XG[p,:] + F*(XG[j,:] - XG[k,:])
           
            for j in range(D):
                if son[j] >xmin[j]  and son[j] < xmax[j]:
                    XG_next_1[i,j] = son[j]
                else:
                    XG_next_1[i,j] = (xmax[j] - xmin[j])*np.random.uniform(size=(1,1)) + xmin[j]
        for i in range(Np):
            randx = np.array(range(D))
            np.random.shuffle(randx)
            for j in range(D):
                if np.random.uniform()>CR and randx[0]!=j:
                     XG_next_2[i,j] = XG[i,j]
                else:
                    XG_next_2[i,j] = XG_next_1[i,j]
       
    ##    CR=CRmax-G*(CRmax-CRmin)/Gm 
        for i in range(Np):
            if f(XG_next_2[i,:]) < f(XG[i,:]):
                XG_next[i,:] = XG_next_2[i,:]
            else:
                XG_next[i,:] = XG[i,:]
            
        for i in range(Np):
            value[0,i] = f(XG_next[i,:])
        value_min =  min(value[0,:])
        pos_min = value[0,:].tolist().index(value_min) 
        
        Gmin[0,G] = value_min
        best_x[G,:] = XG_next[pos_min,:]
      
        XG = XG_next
        trace[G,0] = G
        trace[G,1] = value_min
        G = G + 1
        
        print u'第%d代进化' % G

    value_min = min(Gmin[0,:])
    pos_min = Gmin[0,:].tolist().index(value_min)
     
    [a,b] = best_x[pos_min,:]

    g = lambda k: (x0[0]-b/a)*np.exp(-a*(k-1))-(x0[0]-b/a)*np.exp(-a*(k-2))
    delta = np.abs(x0 - np.array([g(i) for i in range(1,len(x0)+1)]))
    e = delta.mean()
    C = delta.std()/x0.std()
    P = 1.0*(np.abs(delta - delta.mean()) < 0.6745*x0.std()).sum()/len(x0)
    print e,C,P

    plt.figure(figsize=(8,6))
    plt.plot(trace[:,0],trace[:,1],color='red')
    plt.xlabel(u'进化代数')
    plt.ylabel(u'最优解')
    plt.title(u'DE算法')
    plt.show()
    return g, a, b, x0[0], C, P 
    
    


    

    
   

        
        
            
    

