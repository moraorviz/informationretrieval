import matplotlib.pyplot as plt
from .spearman import Spearman

spearman = Spearman()
x, y = spearman.get_values()
rho, p_value = spearman.calculate(x, y)
print(rho)

plt.scatter(x, y)
plt.show()
