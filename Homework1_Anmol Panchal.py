
# coding: utf-8

# In[1]:


import math 
import pandas 
import numpy as np 
import scipy.stats 
from scipy.stats import uniform
import math
from scipy.stats import norm
import matplotlib    
import matplotlib.pyplot as plt


# In[11]:


#Q.1,2,3 from notes
N=1000 
x = np.random.uniform(0,4,N)
y = (0.1994582095*2.178*np.exp(-x*np.exp(2)/8))
m = np.mean(y) 
s = np.var(y) 
z = (norm.pdf(y)/N)
a= np.mean(z)
b= np.var(z)
print(m,s) 
print(a,b)
plt.plot(z,norm.pdf(y))
print(z)


# In[3]:


#Q.5.3
N = 1000
e =  2.718281
def estimate():
    def g(y):
        return e**(-y/2)*1/2
    xs = g(np.random.uniform(size = N))
    theta_hat = np.mean(xs)
    var = np.var(xs)/N
#     df = pd.DataFrame(theta_hat, var)
    print("Theta_hat:",theta_hat," Variance: ",var)


# In[4]:


def estimate_unif_max():
    def g(y):
        return e**(-y)
    xs = g(np.random.uniform(0, 0.5, N))
    var = np.var(xs)/N
    theta_hat = np.mean(xs)*1/2
#     df = pd.DataFrame(theta_hat, var)
    print("Theta_hat:",theta_hat," Variance: ",var)


# In[5]:


def estimate_exp():
    def g(y):
        return 1/y
    y = np.random.exponential(scale = 1, size = N) 
    var = np.var(y)/N
    theta_hat = np.mean(y)
#     df = pd.DataFrame(theta_hat, var)
    print("Theta_hat:",theta_hat," Variance: ",var)


# In[6]:


estimate()
estimate_unif_max()
estimate_exp()


# In[7]:


#Q.5.13
from scipy.stats import rayleigh
def g(x):
    return (x**2)/(np.sqrt(2*math.pi)*np.exp(-x**2/2))

xs = np.arange(0,10.1,0.1)
ys_g = g(xs)
ys_rayleigh = rayleigh.pdf(xs, 1.5)
ys_norm = norm.pdf(xs, 1.5)
lim =  max(np.r_[ys_g, ys_rayleigh, ys_norm])


# In[8]:


import matplotlib.pyplot as plt
plt.plot(xs, ys_g)
plt.plot(xs, ys_rayleigh)
plt.plot(xs, ys_norm)
plt.ylim(ymin = 0, ymax=1)
plt.xlim(xmin = 0, xmax = 10)
plt.show()


# In[9]:


#Q.5.14
def g(x):
    res = x**2/np.sqrt(2*math.pi)*np.exp(-x**2/2)
    return res

sigma_rayleigh = 1.5
mean = 1.5
n = 10000

def f1(x):
    return rayleigh.pdf(x, sigma_rayleigh)

def f2(x):
    return norm.pdf(x, mean)

def rf1():
    return rayleigh.rvs(sigma_rayleigh, size = n)

def rf2():
    return norm.rvs(mean, size = n)

def is_rayleigh():
    xs = rf1()
    a = g(xs)/f1(xs)
    return np.mean(a)

def is_norm():
    xs = rf2()
    a = g(xs)/f2(xs)
    return np.mean(a)


# In[10]:


theta1 = is_rayleigh()
theta2 = is_norm()
print("Theta1:", theta1, " Theta2:", theta2)


# In[29]:



#Q.5.4
import math 
import pandas 
import numpy as np 
import scipy.stats 
from scipy.stats import uniform
import math
from scipy.stats import norm
import matplotlib    
import matplotlib.pyplot as plt
a=3
b=3
x=np.arange(0,1,0.1)
y=scipy.stats.beta.pdf(x,a,b)
z=scipy.stats.beta.cdf(x,a,b)
print(y)
print(z)
plt.plot(x,y)


# In[26]:





# In[27]:


import numpy as np 
import scipy.stats
import math
import pandas
np.random.seed(seed=10000)

a,b = 3,3
x= np.random.normal(size= a * b).reshape((a,b)) 
x[:, 0] = 1 
print(x[:5, :])
betastar = np.array([0, 1, 0.1])
e = np.random.normal(size=a)
y = np.dot(x, betastar) + e
xpinv = scipy.linalg.pinv2(x) 
betahat = np.dot(xpinv, y)
betahat1 = scipy.stats.beta(xpinv,y)
print("Estimated beta:\n", betahat)
print("Estimated beta:\n", betahat1)

