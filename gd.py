import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])
def compute_gradient(x,y,w,b):
    m = len(x)
    i=0
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb_i = w * x[i] + b
        dj_db_i = f_wb_i - y[i]
        dj_dw_i = (f_wb_i - y[i]) * x[i]
        dj_db = dj_db + dj_db_i
        dj_dw = dj_dw + dj_dw_i
    dj_db = (1 / m) * dj_db
    dj_dw = (1 / m) * dj_dw

    return dj_dw, dj_db

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist)
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("cost vs. iteration"); ax2.set_title("cost vs. iteration (tail)")
ax1.set_ylabel('cost') ; ax2.set_ylabel('cost')
ax1.set_xlabel('iteration step') ; ax2.set_xlabel('iteration step')
plt.show()
fig, ax = plt.subplots(1,1, figsize=(12,6))
plt_contour_wgrad(x_train, y_train,p_hist, ax)
fig, ax = plt.subplots(1,1, figsize=(12,4))
plt_contour_wgrad(x_train,y_train,p_hist,ax,w_range=[180,220,0.5], contours=[1,5,10,20],resolution=0.5)
w_init=0
b_init=0
iterations = 10
tmp_alpha = 8.0e-1
w_final, b_final, J_hist, p_hist = gradient_descent(x_train,y_train,w_init,b_init,tmp_alpha,iterations,compute_cost, compute_gradient)
plt_divergence(p_hist, J_hist, x_train, y_train)
plt.show()
