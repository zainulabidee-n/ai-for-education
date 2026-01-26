# Generate three performance graphs using matplotlib (no colors specified)

import matplotlib.pyplot as plt

models = ["DenseNet121", "MobileNetV2", "EfficientNetB0"]

accuracy = [10.92, 12.1, 9.22]
params = [6964106, 2236682, 4020358]
speed = [7.26, 40.74, 16.48]
memory = [1090.43, 910.51, 886.21]

# Accuracy graph
plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy (%)")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.savefig('accuracy.png')
plt.close()

# Inference Speed graph
plt.figure()
plt.bar(models, speed)
plt.title("Inference Speed (images/sec)")
plt.xlabel("Models")
plt.ylabel("Images/sec")
plt.savefig('speed.png')
plt.close()

# Memory Usage graph
plt.figure()
plt.bar(models, memory)
plt.title("Memory Usage (MB)")
plt.xlabel("Models")
plt.ylabel("MB")
plt.savefig('memory.png')
plt.close()

print("Graphs saved.")