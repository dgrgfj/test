import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(6, 6.5))

i = np.arange(0, 1000, 1)
j_1 = np.random.randn(1000, 1)
j_2 = np.random.randn(1000, 1)
j_3 = np.random.randn(1000, 1)
pic_1 = plt.subplot(311)
pic_1.set_title("pic_1")
plt.plot(i, j_1)
plt.subplots_adjust(hspace=1)
pic_2 = plt.subplot(312)
pic_2.set_title("pic_2")
plt.plot(i, j_2)
plt.subplots_adjust(hspace=1)
pic_3 = plt.subplot(313)
pic_3.set_title("pic_3")
plt.plot(i, j_3)

plt.show()
