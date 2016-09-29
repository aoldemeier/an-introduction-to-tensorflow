import numpy as np

index = 5

formatted_array = np.array(mnist.test.images[index]).reshape (28,28)
print(formatted_array)
plt.imshow(formatted_array)
print("Correct digit: ", np.argmax(mnist.test.labels[index]))

print("Prediction: ", activations_max_indices[index])
