#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
x = np.array(([2,9],[1,5],[3,6]),dtype=float)
y = np.array(([92],[86],[89]),dtype=float)

x=x/np.amax(x,axis=0)
y=y/100

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_grad(x):
	return x*(1-x)

epoch = 1000		#iterations
eta = 0.2
inp_neu = 2
hid_neu = 3
oup_neu = 1

wh = np.random.uniform(size=(inp_neu,hid_neu))
bh = np.random.uniform(size=(1,hid_neu))

wout = np.random.uniform(size=(hid_neu,oup_neu))
bout = np.random.uniform(size=(1,oup_neu))

for i in range(epoch):
	h_ip = np.dot(x,wh)+bh
	h_act = sigmoid(h_ip)
	hiddengrad = sigmoid_grad(h_act)

	o_ip = np.dot(h_act,wout)+bout
	output = sigmoid(o_ip)
	outgrad = sigmoid_grad(output)

	Eo = y-output
	d_output = Eo*outgrad

	Eh = d_output.dot(wout.T)
	d_hidden = Eh*hiddengrad

	wout +=h_act.T.dot(d_output)*eta
	wh += x.T.dot(d_hidden)*eta

print("Normalized input: ")
print(x)
print("Actual ouput: ")
print(y)
print("Predicted output: ")
print(output)

