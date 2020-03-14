#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests


url = 'http://localhost:5000/results'
r = requests.post(url,json={'VEHICLE_ID':5, 'ENGINE_COOLANT_TEMP':200, 'ENGINE_RPM':400,'AIR_INTAKE_TEMP':40,'THROTTLE_POS':80})


# In[ ]:




