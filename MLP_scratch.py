import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

class MLP():
    def __init__(self):
        self.hiddenLayers = 0
        self.outputLayer = 0
        self.lossLayer = 0
        self.h = []
        self.output = 0
        self.loss = 0
    
    def add_layer(self, ltype, dim_in, dim_out):
        if ltype == "Hidden":
            self.h.append(Hidden(dim_in, dim_out))
            self.hiddenLayers +=1
        elif ltype == "Output":
            self.output = Output(dim_in, dim_out)
            self.outputLayer = 1
        elif ltype =="Loss":
            self.loss = Loss(dim_in, dim_out)
            self.lossLayer = 1
    def forward_pass(self, x, y, alpha, testBool = False):
        for i in range (0, self.hiddenLayers):
            if i == 0:
                self.h[i].hidden_forward_1(x, testBool)
            else:
                self.h[i].hidden_forward(self.h[i-1], testBool)
        self.output.output_forward(self.h[-1], testBool)
        Ws = []
        for i in range(0, len(self.h)):
            Ws.append(self.h[i].W)
        Ws.append(self.output.Wo)
        if testBool == False:
            self.loss.loss_forward(self.output.forward, y, alpha, Ws, testBool)
        else:
            self.loss.loss_forward(self.output.testForward, y, alpha, Ws, testBool)
        if testBool == False:
            return [layer.forward for layer in self.h], self.output.forward, self.loss.forward
        else:
            return self.loss.testForward

    def backward_pass(self, x, y, h, z, alpha):
        self.loss.loss_backward(z, y)
        dLdz = self.loss.dLdz
        self.output.output_backward(self.h[-1].forward, dLdz)
        dLdWo = self.output.dLdWo
        dLdb = dLdz
        dLdh = [0] * self.hiddenLayers
        dLdW = [0] * self.hiddenLayers
        dLdc = [0] * self.hiddenLayers

        for i in range(self.hiddenLayers-1,-1, -1):
            if i == 0:
                prevInput = x
            else:
                prevInput = self.h[i-1].forward
            if(i == self.hiddenLayers-1):
                self.h[i].hidden_backward(dLdz, None, prevInput, self.output.Wo, alpha)
            else:
                self.h[i].hidden_backward(self.h[i+1].dLdh,self.h[i+1].W, prevInput, None, alpha)
            dLdh[i] = self.h[i].dLdh
            dLdW[i] = self.h[i].dLdW
            dLdc[i] = self.h[i].dLdc
        return dLdz, dLdWo, dLdb, dLdh, dLdW, dLdc

    def updateParameters(self,dLdz, dLdWo, dLdb, dLdh, dLdW, dLdc, learn, batch_size, alpha):
        lamb = alpha * 2
        for i in range(0, self.hiddenLayers):
            self.h[i].W = self.h[i].W - (learn/ batch_size) * (dLdW[i]+self.h[i].W*lamb)
            self.h[i].c  = self.h[i].c - (learn/ batch_size) * dLdc[i]
            self.h[i].dLdh = self.h[i].dLdh - (learn / batch_size) * dLdh[i]
        self.output.Wo = self.output.Wo - (learn/ batch_size) * dLdWo
        self.output.b = self.output.b - (learn/ batch_size) * dLdb
        self.loss.dLdz = self.loss.dLdz - (learn/batch_size) * dLdz
    def getParameters(self):
        return self.output.Wo, self.output.b, [h.W for h in self.h], [h.c for h in self.h]

    def train(self, x, y, epochs, batch_size, alpha, test = None, y_test = None): #test and y_test are used in part C to determine early stopping
        rounds = epochs * int((x.shape[0]/batch_size))
        testingLoss = rounds * [random.randint(0, 1)]
        loss = rounds * [random.randint(0, 1)]
        batchNum = 0
        batchNum2 = 0
        Wos = []
        bs = []
        Ws = []
        cs = []
        for rounds in range(1, rounds):
            learn = 0.01
            for sample in range(1, batch_size + 1):
                h, z, loss[rounds] = self.forward_pass(np.transpose(x[batchNum * batch_size + sample,:]), y[batchNum * batch_size + sample], alpha)
                if test is not None and y_test is not None: 
                    testingLoss[rounds] = self.forward_pass(np.transpose(test[batchNum2 * batch_size + sample,:]),  y_test[batchNum2 * batch_size + sample], alpha, True)
                dLdz, dLdWo, dLdb, dLdh, dLdW, dLdc = self.backward_pass(np.transpose(x[batchNum * batch_size + sample,:]), y[batchNum * batch_size + sample], h, z, alpha)
                if sample == 1:
                    cummulative_dLdz = dLdz
                    cummulative_dLdWo = dLdWo
                    cummulative_dLdb = dLdb
                    cummulative_dLdh = dLdh
                    cummulative_dLdW = dLdW
                    cummulative_dLdc = dLdc
                else:
                    cummulative_dLdz += dLdz
                    cummulative_dLdWo += dLdWo
                    cummulative_dLdb += dLdb
                    for i in range(0, len(dLdW)):
                        cummulative_dLdh[i] = np.matrix(cummulative_dLdh[i]) + np.matrix(dLdh[i])
                        cummulative_dLdW[i] = np.matrix(cummulative_dLdW[i]) + np.matrix(dLdW[i])
                        cummulative_dLdc[i] = np.matrix(cummulative_dLdc[i]) + np.matrix(dLdc[i])
                Wos.append(self.output.Wo)
                bs.append(self.output.b)
                Ws.append(self.h[-1].W)
                cs.append(self.h[-1].c)
            batchNum +=1
            if test is not None and y_test is not None: 
                batchNum2 += 1
            if (batch_size * (batchNum + 1) >= x.shape[0]):
                batchNum = 0
            if test is not None and y_test is not None: 
                if (batch_size* (batchNum2 + 1) >= test.shape[0]):
                    batchNum2 = 0
            print(rounds, batchNum)
            self.updateParameters(cummulative_dLdz, cummulative_dLdWo, cummulative_dLdb, cummulative_dLdh, cummulative_dLdW, cummulative_dLdc, learn, batch_size, alpha)
            Wo, b, W, c = self.getParameters()
        if test is not None and y_test is not None:
            return Wos, bs, Ws, cs, loss, testingLoss
        else:
            return Wo, b, W, c, loss

    def predict(self,z):
        P = [0] * z.shape[0]
        denom = 0
        for i in range(0, z.shape[0]):
            denom += np.exp(z[i, 0])
        for i in range(0, z.shape[0]):
            P[i] = np.exp(z[i, 0])/ denom
        return P.index(max(P))
 
    def sampleAndTest(self, alpha):
        X = 5 * np.matrix(np.random.random((10000, 2)))- 2.5
        y = 10000 * [random.randint(0,1)]
        p = np.zeros(10000)
        samples = X.shape[0]
        for i in range(0, samples):
            h, z, a = self.forward_pass(np.transpose(X[i, :]), y[i], alpha)
            p[i] = self.predict(z) # z is output vector (1 x 3)
        return X, p


class Hidden():
    def __init__(self, dim_in, dim_out):
        self.W = 0.1 * np.random.random((dim_in, dim_out))
        self.c = 0.15 *  np.random.random((dim_out, 1))
        self.beforeRelu = 0
        self.forward = 0
        self.dLdh = 0
        self.dLdW = 0
        self.dLdc = 0
        self.testForward = 0
    def hidden_forward_1(self, x, testBool):
        if (testBool == True):
            validBeforeRelu = np.matmul(np.transpose(self.W), x) + self.c
            self.testForward = np.maximum(validBeforeRelu, 0.1 * validBeforeRelu)
        else:
            self.beforeRelu = np.matmul(np.transpose(self.W), x) + self.c
            self.forward = np.maximum(self.beforeRelu, 0.1 * self.beforeRelu)
    def hidden_forward(self, h, testBool):
        if (testBool == True):
            validBeforeRelu = np.matmul(np.transpose(self.W), h.testForward) + self.c
            self.testForward = np.maximum(self.beforeRelu, 0.1 * self.beforeRelu)
        else:
            self.beforeRelu = np.matmul(np.transpose(self.W), h.forward) + self.c
            self.forward = np.maximum(self.beforeRelu, 0.1 * self.beforeRelu)
    def hidden_backward(self,ahead, nextW, prevInput, Wo, alpha):
        if Wo is not None:
            self.dLdh = np.matmul(Wo, ahead)
        else:
            self.dLdh = np.matmul(nextW, ahead)
        f = 1.0 * (self.beforeRelu > 0.0) + 0.01 * (self.beforeRelu < 0.0)
        self.dLdW = np.matmul(prevInput, np.transpose(np.multiply(f, self.dLdh))) + ((4 * alpha)* (self.W))
        self.dLdc = np.multiply(f, self.dLdh)
class Output():
    def __init__(self, dim_in, dim_out):
        self.Wo = np.random.random((dim_in, dim_out))
        self.b = np.random.random((dim_out, 1))
        self.forward = 0
        self.testForward = 0
        self.dLdWo = 0
        self.dLdb = 0
    def output_forward(self, h, testBool):
        if (testBool == True):
            self.testForward = np.dot(np.transpose(self.Wo), h.testForward) + self.b
        else:
            output = np.dot(np.transpose(self.Wo), h.forward)
            self.forward = output + self.b
    def output_backward(self, h, dLdz):
        self.dLdWo = np.matmul(h, np.transpose(dLdz))
        self.dLdb = dLdz

class Loss():
    def __init__(self, dim_in, dim_out):
        self.P = 0
        self.forward = 0
        self.testForward = 0
        self.dLdz = np.zeros((dim_out, 1))
    
    def loss_forward(self, z, y, alpha, Ws, testBool):#todo add regularization
        denom = 0
        for i in range(0, z.shape[0]):
            denom += np.exp(z[i, 0])

        if testBool == True:
            P = np.exp(z[int(y)])/ denom
            norms = [(np.linalg.norm(W, ord = 'fro') ** 2) for W in Ws]
            self.testForward = -np.log(P) + (alpha * sum(norms))
            print(self.testForward)
        else:
            self.P = np.exp(z[int(y)])/denom
            norms = [(np.linalg.norm(W, ord = 'fro') ** 2) for W in Ws]
            self.forward = -np.log(self.P) + (alpha * sum(norms))
    def loss_backward(self, z, y):
        denom = 0
        for i in range(0, z.shape[0]):
            denom += np.exp(z[i, 0])
        for i in range(0, self.dLdz.shape[0]):
            if y == i:
                self.dLdz[i, 0] = -1 + (np.exp(z[i, 0]))/denom
            else:
                self.dLdz[i, 0] = (np.exp(z[i, 0]))/denom
       
def plot_decision(X, y_predict):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    indices_0 = [k for k in range(0, X.shape[0]) if y_predict[k] == 0]
    indices_1 = [k for k in range(0, X.shape[0]) if y_predict[k] == 1]
    indices_2 = [k for k in range(0, X.shape[0]) if y_predict[k] == 2]

    ax.plot(X[indices_0, 0], X[indices_0,1], marker='o', linestyle='', ms=5, label='0')
    ax.plot(X[indices_1, 0], X[indices_1,1], marker='o', linestyle='', ms=5, label='1')
    ax.plot(X[indices_2, 0], X[indices_2,1], marker='o', linestyle='', ms=5, label='2')

    ax.legend()
    ax.legend(loc=2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Tricky 3 Class Classification')
    plt.show()

def earlyStopping(NN, train_losses, test_losses, roundsPerEpoch, Wos, bs, Ws, cs):
    fig, ax = plt.subplots(figsize = (12, 8))
    stoppingPoint = 0
    epochCount = -1
    t = []
    t2 = []
    for i in range(0, len(train_losses)):
        if i % roundsPerEpoch == 0 and i!=0:
            epochCount +=1
            start = epochCount * roundsPerEpoch
            t.append(float(np.mean(train_losses[start:i])))
            t2.append(float(np.mean(test_losses[start:i])))
            if epochCount == 35:
                stoppingPoint = i
    NN.output.Wo = Wos[i] #set weights to values at early stopping point 
    NN.output.b = bs[i]
    NN.h[-1].W = Ws[i]
    NN.h[-1].c = cs[i]
    
    X = [idx+1 for idx, val in enumerate(t)]
    X2 = [idx+1 for idx, val in enumerate(t2)]
    ax.plot(X, t, color = 'red', label = 'Training', ms= 1)
    ax.plot(X2, t2, color = 'blue', label = 'Testing', ms = 1)   
    ax.legend()
    ax.legend(loc = 2)
    ax.set_xlabel(xlabel = "Epoch")
    ax.set_ylabel(ylabel = "Loss")
    ax.set_title("Average Loss per Epoch")
    plt.show()

def plot_loss(losses, roundsPerEpoch):
    l = []
    epochCount = -1
    for i in range(0, len(losses)):
        if i % roundsPerEpoch == 0 and i!= 0:
            epochCount +=1
            start = epochCount * roundsPerEpoch
            l.append(float(np.mean(losses[start:i])))
    X = [idx+1 for idx, val in enumerate(l)]
    plt.plot(X, l, color = "red" )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Loss per Epoch")
    plt.show()
def generate_dem_rep(): #split data 60:20:20 (Training: Testing: Validation)
    df = pd.read_excel('counties_vote_data2 copy.xlsx')
    data = [list(row) for row in df.values]
    random.shuffle(data)
    train = [county[1:-1] for county in data[0: 1880]]
    test = [county[1:-1] for county in data[1880: 2130]]
    validation = [county[1:-1] for county in data[2130: len(data)]]
    
    train = np.matrix(train)
    test = np.matrix(test)
    validation = np.matrix(validation)
    y_train= np.transpose(np.matrix(train))[-1:,]
    y_test= np.transpose(np.matrix(test))[-1:,]
    y_validation = np.transpose(np.matrix(validation))[-1:,]


    for i in range(0, len(train)): #scaling income data down
        train[i][0] = train[i][0]/100000

    for i in range(0, len(test)):
        test[i][0] = test[i][0]/100000

    for i in range(0, len(validation)):
        validation[i][0] = validation[i][0]/100000
    return train, test, validation, np.transpose(y_train), np.transpose(y_test), np.transpose(y_validation)

def classifyDemReps(NN, x, y):
    y = np.zeros(y.shape[0])
    predictions = np.zeros(y.shape[0])
    correct = 0
    for sample in range(0, x.shape[0]):
        h, val, a = NN.forward_pass(np.transpose(x[sample, :]), y[sample], alpha = 0.0, testBool = False)
        predictions[sample] = NN.predict(val)
        if int(predictions[sample]) == y[sample]:
            correct +=1
    accuracy = (correct/y.shape[0]) * 100
    return accuracy


def dem_rep(train, test, validation, y_train, y_test, y_validation):
    NN = MLP()
    NN.add_layer('Hidden', dim_in=3, dim_out=16)
    NN.add_layer('Output', dim_in=16, dim_out=2)
    NN.add_layer('Loss', dim_in=2, dim_out=2)
    batch_size = 10
    Wos, bs, Ws, cs, train_losses, test_losses = NN.train(train, y_train, epochs=50, batch_size=10, alpha=0.0, test = test, y_test = y_test)
    roundsPerEpoch = int(train.shape[0]/batch_size)
    earlyStopping(NN, train_losses, test_losses, roundsPerEpoch, Wos, bs, Ws, cs)
    accuracy = classifyDemReps(NN, validation, y_validation)
    return accuracy

if __name__ == '__main__':
    
    NN = MLP()
    NN.add_layer('Hidden', dim_in=2, dim_out=16)
    NN.add_layer('Hidden', dim_in=16, dim_out=16)
    NN.add_layer('Hidden', dim_in=16, dim_out=16)
    NN.add_layer('Output', dim_in=16, dim_out=3)
    NN.add_layer('Loss', dim_in=3, dim_out=3)
    data = pd.DataFrame(np.zeros((5000, 3)), columns=['x1', 'x2', 'y'])

    # Let's make up some noisy XOR data to use to build our binary classifier
    for i in range(len(data.index)):
        x1 = random.randint(0,1)
        x2 = random.randint(0,1)
        if x1 == 1 and x2 == 0:
            y = 0
        elif x1 == 0 and x2 == 1:
            y = 0
        elif x1 == 0 and x2 == 0:
            y = 1
        else:
            y = 2
        x1 = 1.0 * x1 + 0.20 * np.random.normal()
        x2 = 1.0 * x2 + 0.20 * np.random.normal()
        data.iloc[i,0] = x1
        data.iloc[i,1] = x2
        data.iloc[i,2] = y
    
    for i in range(int(0.25 *len(data.index))):
        k = np.random.randint(len(data.index)-1)  
        data.iloc[k,0] = 1.5 + 0.20 * np.random.normal()
        data.iloc[k,1] = 1.5 + 0.20 * np.random.normal()
        data.iloc[k,2] = 1

    for i in range(int(0.25 *len(data.index))):
        k = np.random.randint(len(data.index)-1)  
        data.iloc[k,0] = 0.5 + 0.20 * np.random.normal()
        data.iloc[k,1] = -0.75 + 0.20 * np.random.normal()
        data.iloc[k,2] = 2
    
    # Now let's normalize this data.
    data.iloc[:,0] = (data.iloc[:,0] - data['x1'].mean()) / data['x1'].std()
    data.iloc[:,1] = (data.iloc[:,1] - data['x2'].mean()) / data['x2'].std()
        
    data.head()

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    # The cost function is expecting numpy matrices so we need to convert X and y before we can use them.  
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    Wo, b, W, c, loss = NN.train(X, y, epochs=1, batch_size=10, alpha=0.0)
    batch_size = 10
    roundsPerEpoch = int(X.shape[0]/batch_size)
    plot_loss(loss, roundsPerEpoch)
    alpha = 0.0
    X, p = NN.sampleAndTest(alpha)
    plot_decision(X, p)

    train, test, validation, y_train, y_test, y_validation = generate_dem_rep()
    dem_rep(train,test, validation, y_train, y_test, y_validation)
