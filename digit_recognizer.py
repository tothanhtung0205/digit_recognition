'''
there are 3 layer
inputlayer : 784 units
hiddenlayer : 30 units
outputlayer : 10 units
'''
import numpy as np
import cv2
from sklearn.externals import joblib


def get_data():
    #get data from csv into matrix
    data = np.genfromtxt('train.csv',delimiter=',')
    y_train = data[1:,[0]]
    x_train = data[1:,1:]
    print "y train:",y_train.shape
    print "x_train:",x_train.shape
    return x_train,y_train
def sigmoid(z):
    return 1/(1+np.exp(-z))
def derivative_of_sigmoid(z) :
    g = sigmoid(z)
    return g*(1-g)
def random_params_initialize(L_in,L_out):
    '''
    :param L_in: so input vao layer
    :param l_out: so output ra khoi layer
    :param epsilon_init :
    :return: matran voi kich thuoc (L_out,1+L_in) cac params duoc khoi tao ngau nhien trong khoang (-epsilon;epsilon)
    '''
    epsilon_init = 0.12
    return np.random.uniform(low = -epsilon_init,high=epsilon_init,size=(L_out,L_in+1))

def unroll_matrix(input,hidden,output): #unrolling params
    theta1 = random_params_initialize(input,hidden)
    theta2 = random_params_initialize(hidden,output)
    print "theta1 :",theta1.shape
    print "theta2 :",theta2.shape
    theta = np.concatenate((theta1.flatten(),theta2.flatten()))
    return theta


def nn_cost_func(theta,x_train,y_train,inputlayer_size,hidden_layer_size,output_layer_size,_lambda):
    m = x_train.shape[0]     #m = 42000
    x_train = np.concatenate((np.ones((m, 1)), x_train), axis=1)  # add column with all values = 1 into x_train
    theta1 = theta[0:(hidden_layer_size * (inputlayer_size + 1))].reshape([hidden_layer_size, inputlayer_size + 1])
    theta2 = theta[(hidden_layer_size * (inputlayer_size + 1)):].reshape([output_layer_size, hidden_layer_size + 1])
    A1 = x_train  # (42000x785)
    Z2 = np.dot(A1, theta1.T)  # (42000x785).(785x30) = (42000x30)
    A2 = sigmoid(Z2)  # (42000x30)
    A2 = np.concatenate((np.ones((m, 1)), A2), axis=1)  # (42000x31) add column (1) before A2
    Z3 = np.dot(A2, theta2.T)  # (42000x31).(31x10) = (42000x10)
    A3 = sigmoid(Z3)
    #cost = np.zeros([m,1])
    #c = np.array(range(1,output_layer_size+1))

    J = 0
    for i in xrange(m):
        for k in xrange(0, output_layer_size):
            yk = np.int(y_train[i] == k)
            cost = -yk * np.log(A3[i, k]) - (1 - yk) * np.log(1 - A3[i, k])
            J += cost
    J /= m

    # Regularized version of cost function
    J += _lambda * ((theta1[:, 1:] ** 2).sum() + (theta2[:, 1:] ** 2).sum()) / (2 * m)
    return  J


def back_propagation(theta,x_train,y_train,inputlayer_size,hidden_layer_size,output_layer_size,_lambda):
    m = x_train.shape[0]                                            #number of training examples (42000)
    x_train = np.concatenate((np.ones((m, 1)), x_train), axis=1)    #add column with all values = 1 into x_train
    theta1 = theta[0:(hidden_layer_size*(inputlayer_size+1))].reshape([hidden_layer_size,inputlayer_size+1])
    theta2 = theta[(hidden_layer_size*(inputlayer_size+1)):].reshape([output_layer_size,hidden_layer_size+1])
                           #
                          #
    A1 = x_train                                                    #(42000x785)
    Z2 = np.dot(A1,theta1.T)                                        #(42000x785).(785x30) = (42000x30)
    A2 = sigmoid(Z2)                                                #(42000x30)
    A2 = np.concatenate((np.ones((m,1)),A2),axis=1)                 #(42000x31) add column (1) before x_train
    Z3 = np.dot(A2,theta2.T)                                        #(42000x31).(31x10) = (42000x10)
    A3 = sigmoid(Z3)                                                #hypotheis with theta
    delta_upcase_1 = np.zeros(theta1.shape)
    delta_upcase_2 = np.zeros(theta2.shape)
    c = np.array(range(0,output_layer_size))                                        #matrix [0,1..10]
    for r in xrange(m):     #change here
#step 1: forward propagation
        y_r = y_train[r]
        a1_r = A1[[r], :]                                            #(1x785)
        z2_r = Z2[[r], :]
        a2_r = A2[[r], :]
        z3_r = Z3[[r], :]
        a3_r = A3[[r], :]
#step 2: compute delta3==>delta2
        delta_3 = a3_r - (c==y_r)                                  #delta3 = h(theta)x - y    (1x10)
        deriv_grad_z2 = np.concatenate((np.ones((1,1)),derivative_of_sigmoid(z2_r)),axis=1)    #derivate g(z2) add 1
                                                                                                # [1 deriv(gz2))]
        delta_2 = (np.dot(delta_3,theta2))*deriv_grad_z2                                         #(1x31)
#step 3: update delta_upcase
        delta_upcase_1 += np.dot((delta_2[:,1:]).T,a1_r)                                          #(30x1).(1x785) = (30x785)
        delta_upcase_2 += np.dot(delta_3.T,a2_r)                                                #(10x1).(1x31) = (10x31)
    delta_upcase_1 /= m
    delta_upcase_2 /= m
    #regulazation
    delta_upcase_1[:,1:] += _lambda*theta1[:,1:]/m
    delta_upcase_2[:,1:] += _lambda*theta2[:,1:]/m
    #unroll
    delta_upcase = np.concatenate((delta_upcase_1.flatten(),delta_upcase_2.flatten()))
    return delta_upcase


def predict(theta1,theta2,x_test):
    m = x_test.shape[0]
    x_test = np.concatenate((np.ones((m, 1)), x_test), axis=1)
    A1 = x_test  # (42000x785)
    Z2 = np.dot(A1, theta1.T)  # (42000x785).(785x30) = (42000x30)
    A2 = sigmoid(Z2)  # (42000x30)
    A2 = np.concatenate((np.ones((m, 1)), A2), axis=1)  # (42000x31) add column (1) before A2
    Z3 = np.dot(A2, theta2.T)  # (42000x31).(31x10) = (42000x10)
    A3 = sigmoid(Z3)
    p = np.argmax(A3, axis=1)
    return p.reshape((m, 1))
def get_test_data():
    data = np.genfromtxt('test.csv', delimiter=',')
    x_test = data[1:,:]
    print x_test.shape
    return x_test

x = get_test_data()
pass

def fit(model):
    input_layer = 784
    hidden_layer = 100
    output_layer = 10
    _lambda = 100

    try:
        theta = joblib.load(model)
        print("Load params completed!")
    except:
        print("getting data.....")
        x_train,y_train = get_data()
        initial_theta1 = random_params_initialize(input_layer,hidden_layer)
        initial_theta2 = random_params_initialize(hidden_layer,output_layer)
        initial_theta = np.concatenate((initial_theta1.flatten(),initial_theta2.flatten()))

        from scipy.optimize import fmin_l_bfgs_b
        print 'Training Neural Network...'
        theta, f, d = fmin_l_bfgs_b(nn_cost_func,initial_theta,
                                  fprime=back_propagation,
                                  args=(x_train, y_train,input_layer,
                                        hidden_layer,
                                        output_layer,
                                         _lambda),
                                  disp=False, maxiter=100)
        #theta,J_history = gradient_descent(initial_theta,x_train,y_train,input_layer,hidden_layer,output_layer,_lambda,0.001,50)
        #print J_history
        joblib.dump(theta,model)

    theta1 = theta[0:(hidden_layer * (input_layer + 1))].reshape([hidden_layer, input_layer + 1])
    theta2 = theta[(hidden_layer * (input_layer + 1)):].reshape([output_layer, hidden_layer + 1])
    return theta1,theta2

def predict_image():
    theta1, theta2 = fit('ann_100.pkl')
    try:
        img_test =  cv2.imread('temp.png',0)
    except:
        print "file not exist"
        return -1
    img_test = cv2.resize(img_test,(28,28))
    vector = img_test.flatten()
    vector = map(lambda x: 255 - x, vector)
    x = np.array([vector])
    pre = predict(theta1,theta2,x)
    return pre[0,0]
# while(True):
#     file_name = raw_input("Enter file name:(q to quit)")
#     if file_name == 'q':
#         break
#     else:
#         file_name = 'mnist/'+file_name
#         x = predict_image(file_name)
#         if x == -1:
#             print "try another file name"
#         else:
#             print "Predict image %d"%x


# x_test = get_test_data()
# predict_nn = predict(theta1,theta2,x_test)
# print("Predict nn .....")
# print predict_nn
# size_test = predict_nn.shape[0]
# imageId = np.array(range(1,predict_nn.shape[0]+1)).reshape(predict_nn.shape[0],1)
# data = np.concatenate((imageId,predict_nn),axis = 1)
# data = data.astype(int)
# with open("resultTR.csv", "wb") as f:
#     f.write(b'ImageId,Label\n')
#     np.savetxt(f, data , fmt='%i', delimiter=",")












'''
def gradient_checking(theta,x_train,y_train,input_layer,hidden_layer,output_layer,_lambda):                 #J???
    checking_grad = np.zeros(theta.shape)
    exl_matrix = np.zeros(theta.shape)
    epxilon = 0.0001
    for i in range(len(theta)):
        print "theta ", theta
        exl_matrix[i] = epxilon
        theta_plus = theta+exl_matrix
        print "theta plus:",theta_plus
        theta_minus = theta - exl_matrix
        print "theta minus:",theta_minus
        J_theta_plus = nn_cost_func(theta_plus,x_train,y_train,input_layer,hidden_layer,output_layer,_lambda)
        J_theta_minus = nn_cost_func(theta_minus,x_train,y_train,input_layer,hidden_layer,output_layer,_lambda)
        checking_grad[i] = (J_theta_plus-J_theta_minus)/(2*epxilon)
        exl_matrix[i] = 0
    return checking_grad
def test_gradient():
    x_train,y_train = get_data()
    x_train = x_train[:1000,340:350]
    y_train = y_train[:1000,:]
    theta1 = random_params_initialize(10,5,0.12)
    theta2 = random_params_initialize(5,3,0.12)
    theta = np.concatenate((theta1.flatten(),theta2.flatten()))
    _lambda = 3
    checking_Delta = gradient_checking(theta,x_train,y_train,10,5,3,_lambda)
    print "checking_Delta:",checking_Delta
    backpro_Delta = back_propagation(theta, x_train, y_train, 10,5,3, _lambda)
    print "Backpro Delta:",backpro_Delta
    diff = np.linalg.norm(checking_Delta - backpro_Delta) / np.linalg.norm(checking_Delta + backpro_Delta)
    print "diff",diff
    return diff
'''



'''
def gradient_descent(theta,x_train,y_train,inputlayer_size,hidden_layer_size,output_layer_size,_lambda,learning_rate,num_iters):
    m = x_train.shape[0]
    J_history = np.zeros(num_iters)
    delta = back_propagation(theta,x_train,y_train,inputlayer_size,hidden_layer_size,output_layer_size,_lambda)
    #//TODO delta1 delta2
    theta1 = theta[0:(hidden_layer * (input_layer + 1))].reshape([hidden_layer, input_layer + 1])
    theta2 = theta[(hidden_layer * (input_layer + 1)):].reshape([output_layer, hidden_layer + 1])
    for i in xrange(num_iters):
        theta1[:,0]-=learning_rate*delta1[:,0]
        theta1[:,1:]-=learning_rate*(delta1[:,1:] + _lambda*theta1[:,1:]/m)
        theta2[:,0] -= learning_rate*delta2[:,0]
        theta2[:,1:] -= learning_rate*(delta2[:,1:]+_lambda*theta2[:,1:]/m)
        last_theta = np.concatenate((theta1.flatten(),theta2.flatten()))
        J_history[i] = nn_cost_func(last_theta,x_train,y_train,inputlayer_size,hidden_layer_size,output_layer_size,_lambda)
    return  theta, J_history
'''











