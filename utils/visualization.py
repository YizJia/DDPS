import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

def visualization_attention(atten_s, atten_c=None):
    attention_s = atten_s.data.cpu().numpy()
    plt.figure()
    for i in range(len(attention_s)):
        plt.imshow(attention_s[i,:,:],cmap='jet')
        plt.axis("off")
        plt.show()