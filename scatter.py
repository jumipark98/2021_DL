
import numpy as np
samples =  np.array([[1,2],[1,3],[1,4],[2,1],[3,1],[3,3]]).T
mean_x = np.mean(samples[0,:])
mean_y = np.mean(samples[1,:])
print(mean_x,mean_y)

mean_vector = np.array([[mean_x],[mean_y]])
scatter_matrix = np.zeros((2,2))
for i in range(samples.shape[1]):
    scatter_matrix += (samples[:,i].reshape(2,1) - mean_vector).dot((samples[:,i].reshape(2,1) - mean_vector).T)
print(scatter_matrix)
cov1 = np.cov(samples)
print(cov1)
samples2 =  np.array([[2,2],[3,2],[3,4],[5,2],[4,5],[5,4]]).T
mean_x1 = np.mean(samples2[0,:])
mean_y1 = np.mean(samples2[1,:])
print(mean_x1,mean_y1)

mean_vector1 = np.array([[mean_x1],[mean_y1]])
scatter_matrix1 = np.zeros((2,2))
for i in range(samples2.shape[1]):
    scatter_matrix1 += (samples2[:,i].reshape(2,1) - mean_vector1).dot((samples2[:,i].reshape(2,1) - mean_vector1).T)
print(scatter_matrix1)
cov2 = np.cov(samples2)
print(cov2)
S = scatter_matrix + scatter_matrix1
print(S)
inv_S = np.linalg.inv(S)
print('inv',inv_S)
v= inv_S*(np.array([1.83-3.66,2.3-3.16]))
print('V',v)
v = inv_S*(mean_vector.T-mean_vector1.T)
print('v', v)
v = np.array([-0.14,-0.05])
Y1 = (v.T).dot(samples)
Y2 = (v.T).dot(samples2)

print(Y1)
print(Y2)