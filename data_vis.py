import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_excel('testscore.xlsx', sheet_name='Sheet1',
                       engine='openpyxl')
data = np.array(data)

epoch = range(1, 200 + 1)
losses = data[:, 0::2]
scores = data[:, 1::2]

fig = plt.figure(
    figsize=(6, 8)
)

## test loss
plt.subplot(2,1,1)
plt.plot(epoch, losses[:, 0], '--', label='#1')
plt.plot(epoch, losses[:, 1], '--', label='#2')
# ax1.plot(epoch, losses[:, 2], '-', label='Res-DualNet')
plt.plot(epoch, losses[:, 3], '--', label='#3')
plt.plot(epoch, losses[:, 4], 'k-', label='#4 (proposed)')
# ax1.tick_params('y', colors='blue')
plt.legend(fontsize=16, loc='center right')
plt.xlabel("Epoch", fontsize=15)
plt.ylabel('Train Loss', fontsize=15)
plt.grid(True)

## test acc
plt.subplot(2,1,2)
plt.plot(epoch, scores[:, 0], '--', label='#1')
plt.plot(epoch, scores[:, 1], '--', label='#2')
# ax2.plot(epoch, scores[:, 2], '--', label='Res-DualNet')
plt.plot(epoch, scores[:, 3], '--', label='#3')
plt.plot(epoch, scores[:, 4], 'k-', label='#4 (proposed)')
plt.ylabel('Test Accuracy (%)', fontsize=15)
# ax2.tick_params('y', colors='red')

plt.xlabel("Epoch", fontsize=15)
plt.tight_layout()
plt.grid(True)
plt.legend(fontsize=16, loc='center right')
# plt.savefig('../../assets/images/markdown_img/180618_1735_twinx.svg')
plt.show()

# for i in range(losses.shape[1]):
#     plt.plot(losses[:, i])
