#!/usr/bin/env python3

import time, re

max_looper = 1000
t = time.time()

emails = []

for i in range(1, max_looper):
    emails.append("name."+str(i)+".last@gmail.com")
    
printed = False
rex = r'^name\.(999)\.([\w]{0,1}[cCbBaA](s7|st|sT))\@([yo0Og](ahoo|utlook|mail)).com$'
#rex = r'.*999.*'

count = 0
for i in range(0, max_looper):
    for email in emails:
        if(re.match(rex,email)):
            # if(printed == False):
            #     print("email found: " + email)
            #     printed = True
            count += 1
        
print(str(count) + " python3: " + str(time.time() - t))        


