# Import our modules that we are using
import matplotlib.pyplot as plt
import numpy as np

# Create the vectors X and Y
x = np.array(range(10))
y1 = 2*x + (2*x^3)/6+(2*x^5)/120 + (2*x^7)/5040 + (2*x^9)/362880 +  (2*x^11)/39916800 + (2*x^13)/39916800
y2 = 2+(2*x^2)/2+(2*x^4)/ 24 + (2*x^6)/720 + (2*x^8)/40320  + (2*x^10)/3628800 
y = y1/y2
print(y)
    
    
y3= np.array([1,2,3,4,5,6,7,8,9,10])
x3=(np.exp(x) + np.exp(-x))/(np.exp(x) - np.exp(-x))
plt.plot(x3,y3)
plt.show()
   



# Create the plot
plt.plot(x,y,label='y = tanh')
plt.plot(x3,y3)
#plt.plot(x,(1/2)* y + 7,label='y = (1/2) * (x**2) + 7')
#plt.plot(x,y + 3,label='y = x**2 + 3')
#plt.plot(x,y - 5,label='y = x**2 - 5')
#plt.plot(x,y - 3,label='y = x**2 - 3')

# Add a title
plt.title('Tanh')

# Add X and y Label
plt.xlabel('x axis')
plt.ylabel('y axis')

# Add a grid
plt.grid(alpha=.4,linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()