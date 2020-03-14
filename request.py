#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests


url = 'http://127.0.0.1:5000/result_api'
r = requests.post(url,json={'VEHICLE_ID':5, 'ENGINE_COOLANT_TEMP':200, 'ENGINE_RPM':400,'AIR_INTAKE_TEMP':40,'THROTTLE_POS':80})

print(r.json())


# In[ ]:




