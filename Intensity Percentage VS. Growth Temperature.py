import numpy as np 
import matplotlib.pyplot as plt


I0 = np.array([0.1275414149142908, 0.3352984774846262,
               0.3531130139513853, 0.5389361865088192,
               0.7757009350318745, 0.7879399201493192,
               0.7276082323046661,0.7883570129916547])
I30 = np.array([0.8724585850857092, 0.6647015225153737,
               0.6468869860486147, 0.46106381349118064,
               0.2242990649681254, 0.21206007985068073,
               0.27239176769533385,  0.21164298700834525])
GT = ["351~354C ","411~418C","485~525C","525~565C",
       "625~635C","635~640C","647~652C","665~680C"]

plt.plot(GT, I0,'--o', GT, I30,'--o') 
plt.title("Intensity Percentage VS. Growth Temperature", fontsize=25) 
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.xlabel('Growth Temperature', loc ="center", fontsize=18)
plt.ylabel('Intensity Percentage', loc ="center", fontsize=18)
plt.legend(["0-degree Percentage", "30-degree Percentage"], loc ="upper right", fontsize=18) 
plt.show()
