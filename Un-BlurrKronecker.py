#%% Initialisation
import numpy as np
import cv2 

#%%Bonus
def filtreVintage(F):
    Image=np.zeros((512,512,3))
    Image[:,:,0]=0.5*F
    Image[:,:,1]=1*F
    Image[:,:,2]=1.5*F
    cv2.imshow('Vintage',cv2.convertScaleAbs(Image))
    
def filtreFrance(F):
    Image2=np.zeros((512,512,3))
    
    for i in range(341):
        Image2[:,i,0]=F[:,i]
    
    for i in range(172,341):
        Image2[:,i,1]=F[:,i]
        
    for i in range(172,512):   
        Image2[:,i,2]=F[:,i]
    cv2.imshow('France',cv2.convertScaleAbs(Image2))
    
    
#%%Displaying the blurry image
G=np.load('G.npy')
cv2.imshow('image_flou',cv2.convertScaleAbs(G))

lG,cG=np.shape(G)   #% dimension de G
#%% Matrix T from the T.npy file
T=np.load('T.npy')
lT,cT=np.shape(T)      #% dimension de T


#%% Creation of the matrix Atilde

k=9

m2,n2=2**k,2**k
m1,n1=lT//m2,cT//n2

Atilde=np.zeros((m1*n1,m2*n2))

l=0 

for j in range(0,n1) :
    for i in range(0,m1) :
        
        a=T[i*m2:(i+1)*m2,j*n2:(j+1)*n2]
        a=a.reshape((-1,1),order='F')
        
        Atilde[l,:]=a.T
        
        l+=1 
        

        
#%%SVD of Atilde

U,S,V=np.linalg.svd(Atilde,False)

#%% Creation of B & C

vecB=np.sqrt(S[0])*U[:,0] #d'apreès la question 5
vecC=np.sqrt(S[0])*V[0,:]

B=np.reshape(vecB,(m1,n1),order='F')
C=np.reshape(vecC,(m2,n2),order='F')

#%% SVD of B & C
Ub,Sb,VTb=np.linalg.svd(B)
Uc,Sc,VTc=np.linalg.svd(C)

#%% Calculation of the pseudo-inverse of T

Btil=np.zeros((m1,n1), dtype='f')
Ctil=np.zeros((m2,n2), dtype='f')

n=13

for i in range (0,np.shape(Sb)[0]):
    if Sb[i]>=10**(-n):
        Btil[i,i]=1/Sb[i]
    else: 
        Btil[i,i]=0
    
for i in range (0, np.shape(Sc)[0]):
    if Sc[i]>=10**(-n):
        Ctil[i,i]=1/Sc[i]
    else: 
        Ctil[i,i]=0

Ttil=np.kron(VTb.T,VTc.T)@np.kron(Btil,Ctil)@np.kron(Ub.T,Uc.T)

#%% Calculation of the un-blurred image (F).

F=np.zeros(np.shape(G))

for i in range(np.shape(G)[0]):
    F[:,i][np.newaxis]=(Ttil@G[:,i][np.newaxis].T).T


#%% Display of the un-blurred F image

cv2.imshow('defloutage avec approximation Kronecker',cv2.convertScaleAbs(F))

cv2.imwrite('image net.jpg',cv2.convertScaleAbs(F))
#%% Bonus : apply filter and have fun

filtreVintage(F)
filtreFrance(F)