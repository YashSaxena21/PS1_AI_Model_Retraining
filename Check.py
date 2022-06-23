#!/usr/bin/env python
# coding: utf-8

# In[3]:


programfile = open(r'C:\Users\yashs\Desktop\PS1_AI_Model_Retraining.py','r')	#connecting to the code file
code = programfile.read()				#reading the code file

if 'keras' or 'tensorflow' in code:			#because keras or tensorflow keyword is a must for a deep learning program
	print('OK')
else:
	print('NOT A NEURAL NETWORK')


# In[ ]:




