import numpy as np
import matplotlib.pyplot as plt

index = 23

formatted_array = np.array(mnist.test.images[index]).reshape (28,28)
plt.imshow(formatted_array)
print("Correct digit: ", np.argmax(mnist.test.labels[index]))

print("Prediction: ", predictions_evaluated[index])
