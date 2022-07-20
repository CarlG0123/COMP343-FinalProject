import matplotlib as plt

accuracies = []
ITERATIONS = [5,10,20,30,50,100]

plt.plot(ITERATIONS, accuracies)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
plt.show()