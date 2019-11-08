import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1,3,1)
print(x)
y = np.array([2, 4])
print(y)
z1 = np.polyfit(x,y,1)
print(z1)
p1 = np.poly1d(z1)
print(p1)
plt.plot(x,y)
plt.show()