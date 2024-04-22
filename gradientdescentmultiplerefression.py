import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])
def predict_single_loop(x,w,b):
    n=len(x)
    p=0
    for i in range(n):
        p_i=x[i] *w[i]
        p=p+p_i
    p=p+b
    return (p)

x_vec = x_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value:{x_vec}")
f_wb = predict_single_loop(x_vec,w_init,b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
def predict(x,w,b):
    p=np.dot(x,w)+b
    return(p)

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0.0
    for i in range(m):
        f_wb_i = np.dot(x[i],w)+b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2*m)
    return(np.squeeze(cost))

def compute_gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_db,dj_dw

tmp_dj_db, tmp_dj_dw = compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: {tmp_dj_dw}')

def gradient_descent(x,y,w_in,b_in, cost_function, gradient_function, alpha, num_iters):
    m=len(x)
    J_history = []
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if i<100000:
            J_history.append(cost_function(x,y,w,b))
        if i% math.ceil(num_iters/10) == 0:
            print(f"iteration{i:4d}: cost {J_history[-1]:8.2f}")
    return w, b, J_history
    

initial_w=np.zero_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7
w_final, b_final, J_hist = gradient_descent(x_train,y_train,initial_w,initial_b,compute_cost,compute_gradient, alpha,iterations)
print(f"b,w found by gradient descent:{b_final:0.2f},{w_final}")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i],w_final)+b_final:0.2f},target value:{y_train[i]}")

a=np.arange(10)
print(a)

print(f"a[2].shape: {a[2].shape} a[2] ={a[2]},accessing an element returns a scalar")
print(f"a[-1] = {a[-1]}")
try:
    c = a[10]
except Exception as e:
    print ("the error message you'll see is:")
    print(e)

    
fig,(ax1,ax2)=plt.subplots(1,2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist)
ax2.plot(100 +np.arange(len(J_hist[100:])),J_hist[100:])
ax1.set_title("cost vs. iteration"); ax2.set_title("cost vs.iteration(tail)")
ax1.set_ylabel('cost'); ax2.set_ylabel('cost')
ax1.set_xlabel('iteration step'); ax2.set_xlabel('iteration step')
plt.show()
