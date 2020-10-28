import telebot
import numpy as np
import codecs
import sys,random,math
from collections import Counter

np.random.seed(0)

#-------------------------------------------------------------------------------------------------
#START OF FRAMEWORK
class Tensor (object):
    
    def __init__(self,data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None

        if(id is None):
            self.id = np.random.randint(0,1000000000)
        else:
            self.id = id
        
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}
        
        if(creators is not None):
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            if(cnt != 0):
                return False
        return True 
        
    def backward(self,grad=None, grad_origin=None):
        if(self.autograd):
 
            if(grad is None):
                grad = Tensor(np.ones_like(self.data))

            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    return
                    print(self.id)
                    print(self.creation_op)
                    print(len(self.creators))
                    for c in self.creators:
                        print(c.creation_op)
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad
            
            # grads must not have grads of their own
            assert grad.autograd == False
            
            # only continue backpropping if there's something to
            # backprop into and if all gradients (from children)
            # are accounted for override waiting for children if
            # "backprop" was called on this variable directly
            if(self.creators is not None and 
               (self.all_children_grads_accounted_for() or 
                grad_origin is None)):

                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                if(self.creation_op == "sub"):
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if(self.creation_op == "mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new , self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)                    
                    
                if(self.creation_op == "mm"):
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)
                    
                if(self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())

                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                    
                if(self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__())
                    
                if(self.creation_op == "sigmoid"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))
                
                if(self.creation_op == "tanh"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))
                
                if(self.creation_op == "index_select"):
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))
                    
                if(self.creation_op == "cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))
                    
    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)
    
    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="mul")
        return Tensor(self.data * other.data)    

    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim,copies):

        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)
    
    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        
        return Tensor(self.data.transpose())
    
    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self,x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))
    
    def sigmoid(self):
        if(self.autograd):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if(self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))
    
    def index_select(self, indices):

        if(self.autograd):
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])
    
    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        return softmax_output
    
    def cross_entropy(self, target_indices):

        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t),-1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()
    
        if(self.autograd):
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)
        
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())  

class Layer(object):
    
    def __init__(self):
        self.parameters = list()
        
    def get_parameters(self):
        return self.parameters

    
class SGD(object):
    
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha
    
    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0
        
    def step(self, zero=True):
        
        for p in self.parameters:
            
            p.data -= p.grad.data * self.alpha
            
            if(zero):
                p.grad.data *= 0

class Linear(Layer):#ok
    def __init__(self, n_inputs, n_outputs, bias = True, path = None, path_to_hidden = None):
        super().__init__()
        
        self.use_bias = bias
        
        if(path == None):
            W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / (n_inputs))
        else:
            W = np.load(path)
        
        self.weight = Tensor(W, autograd = True)
        if(self.use_bias):
            if(path_to_hidden == None):
                W = np.zeros(n_outputs)
            else:
                W = np.load(path_to_hidden)
            
            self.bias = Tensor(W, autograd = True)

        self.parameters.append(self.weight)
        
        if(self.use_bias):
            self.parameters.append(self.bias)
        
    def forward(self, input):
        if(self.use_bias):
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        return input.mm(self.weight)

class Sequential(Layer):
    
    def __init__(self, layers=list()):
        super().__init__()
        
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


class Embedding(Layer):#ok
    
    def __init__(self, vocab_size, dim, path = None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        
        #reading matrix
        if(path != None):
            W = np.load(path)
        else:
            W = (np.random.rand(vocab_size, dim) - 0.5) / dim
        
        self.weight = Tensor(W, autograd = True)
        
        self.parameters.append(self.weight)
    
    def forward(self, input):
        return self.weight.index_select(input)


class Tanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()
    

class CrossEntropyLoss(object):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return input.cross_entropy(target)

    
class RNNCell(Layer):
    
    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        if(activation == 'sigmoid'):
            self.activation = Sigmoid()
        elif(activation == 'tanh'):
            self.activation == Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)
        
        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()        
    
    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden
    
    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)
    
class LSTMCell(Layer):#ok
    
    def __init__(self, n_inputs, n_hidden, n_output, firstTime = True):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        if(firstTime):
            self.xf = Linear(n_inputs, n_hidden)
            self.xi = Linear(n_inputs, n_hidden)
            self.xo = Linear(n_inputs, n_hidden)        
            self.xc = Linear(n_inputs, n_hidden)        
        
            self.hf = Linear(n_hidden, n_hidden, bias=False)
            self.hi = Linear(n_hidden, n_hidden, bias=False)
            self.ho = Linear(n_hidden, n_hidden, bias=False)
            self.hc = Linear(n_hidden, n_hidden, bias=False)        
        
            self.w_ho = Linear(n_hidden, n_output, bias=False)
        else:
            self.xf = Linear(n_inputs, n_hidden, path = 'xf.npy', path_to_hidden = 'xf_bias.npy')
            self.xi = Linear(n_inputs, n_hidden, path = 'xi.npy', path_to_hidden = 'xi_bias.npy')
            self.xo = Linear(n_inputs, n_hidden, path = 'xo.npy', path_to_hidden = 'xo_bias.npy')
            self.xc = Linear(n_inputs, n_hidden, path = 'xc.npy', path_to_hidden = 'xc_bias.npy')
        
            self.hf = Linear(n_hidden, n_hidden, bias = False, path = 'hf.npy')
            self.hi = Linear(n_hidden, n_hidden, bias = False, path = 'hi.npy')
            self.ho = Linear(n_hidden, n_hidden, bias = False, path = 'ho.npy')
            self.hc = Linear(n_hidden, n_hidden, bias = False, path = 'hc.npy')
        
            self.w_ho = Linear(n_hidden, n_output, bias = False, path = 'w_ho.npy')
        
        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()

        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()        
        self.parameters += self.ho.get_parameters()        
        self.parameters += self.hc.get_parameters()                
        
        self.parameters += self.w_ho.get_parameters()        
    
    def forward(self, input, hidden):
        
        prev_hidden = hidden[0]        
        prev_cell = hidden[1]
        
        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()        
        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()        
        c = (f * prev_cell) + (i * g)

        h = o * c.tanh()
        
        output = self.w_ho.forward(h)
        return output, (h, c)
    
    def init_hidden(self, batch_size=1):
        init_hidden = Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)
        init_cell = Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)
        init_hidden.data[:,0] += 1
        init_cell.data[:,0] += 1
        return (init_hidden, init_cell)
#END of FRAMEWORK
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
#SAVING FUNCTIONS
def save_ALL():
    np.save('xf', np.array(model.xf.weight.data))
    np.save('xi', np.array(model.xi.weight.data))
    np.save('xo', np.array(model.xo.weight.data))
    np.save('xc', np.array(model.xc.weight.data))

    np.save('xf_bias', np.array(model.xf.bias.data))
    np.save('xi_bias', np.array(model.xi.bias.data))
    np.save('xo_bias', np.array(model.xo.bias.data))
    np.save('xc_bias', np.array(model.xc.bias.data))

    np.save('hf', np.array(model.hf.weight.data))
    np.save('hi', np.array(model.hi.weight.data))
    np.save('ho', np.array(model.ho.weight.data))
    np.save('hc', np.array(model.hc.weight.data))

    np.save('w_ho', np.array(model.w_ho.weight.data))

    np.save('embed', np.array(embed.weight.data))

    np.save('indices', indices)

#writing vocabulary
def write_vocab():
    with codecs.open("vocab.txt", 'w',"utf-8") as vocabulary_file:
        for listitem in vocab:
            if(listitem != '\n'):
                vocabulary_file.write('%s\n' % listitem)
            else:
                vocabulary_file.write('%s' % listitem)
#END OF SAVING FUNCTIONS
#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
#TRANSLIT FUNCTIONS

def to_lang(s = ""):
    st = ""
    for el in s:
        if el == 'а':
            st += 'a'
        elif el == 'б':
            st += 'b'
        elif el == 'в':
            st += 'v'
        elif el == 'г':
            st += 'g'
        elif el == 'д':
            st += 'd'
        elif el == 'е':
            st += 'e'
        elif el == 'ё':
            st += 'yo'
        elif el == 'ж':
            st += 'zh'
        elif el == 'з':
            st += 'z'
        elif el == 'и':
            st += 'i'
        elif el == 'й':
            st += 'j'
        elif el == 'к':
            st += 'k'
        elif el == 'л':
            st += 'l'
        elif el == 'м':
            st += 'm'
        elif el == 'н':
            st += 'n'
        elif el == 'о':
            st += 'o'
        elif el == 'п':
            st += 'p'
        elif el == 'р':
            st += 'r'
        elif el == 'с':
            st += 's'
        elif el == 'т':
            st += 't'
        elif el == 'у':
            st += 'u'
        elif el == 'ф':
            st += 'f'
        elif el == 'х':
            st += 'h'
        elif el == 'ц':
            st += 'c'
        elif el == 'ч':
            st += 'ch'
        elif el == 'ш':
            st += 'sh'
        elif el == 'щ':
            st += 'sh\''
        elif el == 'ъ':
            st += '`'
        elif el == 'ы':
            st += 'y'
        elif el == 'ь':
            st += '\''
        elif el == 'э':
            st += 'eh'
        elif el == 'ю':
            st += 'yu'
        elif el == 'я':
            st += 'ja'
        elif el == 'А':
            st += 'A'
        elif el == 'Б':
            st += 'B'
        elif el == 'В':
            st += 'V'
        elif el == 'Г':
            st += 'G'
        elif el == 'Д':
            st += 'D'
        elif el == 'Е':
            st += 'E'
        elif el == 'Ё':
            st += 'YO'
        elif el == 'Ж':
            st += 'ZH'
        elif el == 'З':
            st += 'Z'
        elif el == 'И':
            st += 'I'
        elif el == 'Й':
            st += 'J'
        elif el == 'К':
            st += 'K'
        elif el == 'Л':
            st += 'L'
        elif el == 'М':
            st += 'M'
        elif el == 'Н':
            st += 'N'
        elif el == 'О':
            st += 'O'
        elif el == 'П':
            st += 'P'
        elif el == 'Р':
            st += 'R'
        elif el == 'С':
            st += 'S'
        elif el == 'Т':
            st += 'T'
        elif el == 'У':
            st += 'U'
        elif el == 'Ф':
            st += 'F'
        elif el == 'Х':
            st += 'H'
        elif el == 'Ц':
            st += 'C'
        elif el == 'Ч':
            st += 'CH'
        elif el == 'Ш':
            st += 'SH'
        elif el == 'Щ':
            st += 'SH\''
        elif el == 'Ъ':
            st += '`'
        elif el == 'Ы':
            st += 'Y'
        elif el == 'Ь':
            st += '\''
        elif el == 'Э':
            st += 'EH'
        elif el == 'Ю':
            st += 'YU'
        elif el == 'Я':
            st += 'JA'
        else:
            st += el
    return st

def back_lang(s = ""):
    st = ""
    l = int(len(s))
    i = 0
    while i < l:
        el = s[i]
        if el == 'a':
            st += 'а'
        elif el == 'b':
            st += 'б'
        elif el == 'v':
            st += 'в'
        elif el == 'g':
            st += 'г'
        elif el == 'd':
            st += 'д'
        elif el == 'e':
            if i+1 < l and s[i+1] == 'h':
                i+=1
                st += 'э'
            else:
                st += 'е'
        elif el == 'z':
            if i+1 < l and s[i+1] == 'h':
                i+=1
                st += 'ж'
            else:
                st += 'з'
        elif el == 'i':
            st += 'и'
        elif el == 'j':
            if i+1 < l and s[i+1] == 'a':
                i+=1
                st += 'я'
            else:
                st += 'й'
        elif el == 'k':
            st += 'к'
        elif el == 'l':
            st += 'л'
        elif el == 'm':
            st += 'м'
        elif el == 'n':
            st += 'н'
        elif el == 'o':
            st += 'о'
        elif el == 'p':
            st += 'п'
        elif el == 'r':
            st += 'р'
        elif el == 's':
            if i+1 < l and s[i+1] == 'h':
                i+=1
                
                if i+1 < l and s[i+1] == '\'':
                    i+=1
                    st += 'щ'
                else:
                    st += 'ш'
            else:
                st += 'с'
        elif el == 't':
            st += 'т'
        elif el == 'u':
            st += 'y'
        elif el == 'f':
            st += 'ф'
        elif el == 'h':
            st += 'х'
        elif el == 'c':
            if i+1 < l and s[i+1] == 'h':
                i+=1
                st += 'ч'
            else:
                st += 'ц'
        elif el == '`':
            st += 'ъ'
        elif el == 'y':
            if i+1 < l and s[i+1] == 'o':
                i+=1
                st += 'ё'
            elif i+1 < l and s[i+1] == 'u':
                i+=1
                st += 'ю'
            else:
                st += 'ы'
        elif el == '\'':
            st += 'ь'
        elif el == 'A':
            st += 'А'
        elif el == 'B':
            st += 'Б'
        elif el == 'V':
            st += 'В'
        elif el == 'G':
            st += 'Г'
        elif el == 'D':
            st += 'Д'
        elif el == 'E':
            if i+1 < l and s[i+1] == 'H':
                i+=1
                st += 'Э'
            else:
                st += 'Е'
        elif el == 'Z':
            if i+1 < l and s[i+1] == 'H':
                i+=1
                st += 'Ж'
            else:
                st += 'З'
        elif el == 'Z':
            st += 'З'
        elif el == 'I':
            st += 'И'
        elif el == 'J':
            if i+1 < l and s[i+1] == 'A':
                i+=1
                st += 'Я'
            else:
                st += 'Й'
        elif el == 'K':
            st += 'К'
        elif el == 'L':
            st += 'Л'
        elif el == 'M':
            st += 'М'
        elif el == 'N':
            st += 'Н'
        elif el == 'O':
            st += 'О'
        elif el == 'P':
            st += 'П'
        elif el == 'R':
            st += 'Р'
        elif el == 'S':
            if i+1 < l and s[i+1] == 'H':
                i+=1
                
                if i+1 < l and s[i+1] == '\'':
                    i+=1
                    st += 'Щ'
                else:
                    st += 'Ш'
            else:
                st += 'С'
        elif el == 'T':
            st += 'Т'
        elif el == 'U':
            st += 'У'
        elif el == 'F':
            st += 'Ф'
        elif el == 'H':
            st += 'Х'
        elif el == 'C':
            if i+1 < l and s[i+1] == 'H':
                i+=1
                st += 'Ч'
            else:
                st += 'C'
        elif el == 'Y':
            if i+1 < l and s[i+1] == 'O':
                i+=1
                st += 'Ё'
            elif i+1 < l and s[i+1] == 'U':
                i+=1
                st += 'Ю'
            else:
                st += 'Ы'
        else:
            st += el
        i+=1
    return st

#END OF TRANSLIT FUNCTIONS
#-----------------------------------------------------------------------------------------------------

firstly = False #TRUE only then training

if(firstly):
    vocab = list(set(raw))
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word]=i
    indices = np.array(list(map(lambda x:word2index[x], raw)))

    embed = Embedding(vocab_size=len(vocab),dim=512)
    model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
    model.w_ho.weight.data *= 0

    save_ALL()
    write_vocab()
else:
    #start of vocabulary rading
    vocab = []

    with open('vocab.txt', 'r') as vocabulary_file:
        filecontent = vocabulary_file.readlines()

        for line in filecontent:
            if(line == '\n'):
                cur_place = '\n'
            else:
                cur_place = line[:-1]

            vocab.append(cur_place)
    #end of vocabulary reading
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i

    indices = np.load('indices.npy')

    embed = Embedding(vocab_size = len(vocab), dim = 512, path = 'embed.npy')
    model = LSTMCell(n_inputs = 512, n_hidden = 512, n_output = len(vocab), firstTime = False)

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)

#-------------------------------------------------------------------------------------------------------
#MAIN FUNCTION
def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)

        m = output.data.argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s
#------------------------------------------------------------------------------------------------------

batch_size = 16
bptt = 25
n_batches = int((indices.shape[0] / (batch_size)))

trimmed_indices = indices[:n_batches*batch_size]
batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()

input_batched_indices = batched_indices[0:-1]
target_batched_indices = batched_indices[1:]

n_bptt = int(((n_batches-1) / bptt))
input_batches = input_batched_indices[:n_bptt*bptt].reshape(n_bptt,bptt,batch_size)
target_batches = target_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
min_loss = 1000
#-----------------------------------------------------------------------------------------------------
#TRAINING FUNCTION
def train(iterations=400, save=False):
    for iter in range(iterations):
        total_loss = 0
        n_loss = 0

        hidden = model.init_hidden(batch_size=batch_size)
        batches_to_train = len(input_batches)

        for batch_i in range(batches_to_train):

            hidden = (Tensor(hidden[0].data, autograd=True), Tensor(hidden[1].data, autograd=True))

            losses = list()
            for t in range(bptt):
                input = Tensor(input_batches[batch_i][t], autograd=True)
                rnn_input = embed.forward(input=input)
                output, hidden = model.forward(input=rnn_input, hidden=hidden)

                target = Tensor(target_batches[batch_i][t], autograd=True)    
                batch_loss = criterion.forward(output, target)

                if(t == 0):
                    losses.append(batch_loss)
                else:
                    losses.append(batch_loss + losses[-1])

            loss = losses[-1]

            loss.backward()
            optim.step()
            total_loss += loss.data / bptt
            
            global min_loss
            epoch_loss = np.exp(total_loss / (batch_i+1))
            if(epoch_loss < min_loss):
                min_loss = epoch_loss
                print()

            log = "\r Iter:" + str(iter)
            log += " - Alpha:" + str(optim.alpha)[0:5]
            log += " - Batch "+str(batch_i+1)+"/"+str(len(input_batches))
            log += " - Min Loss:" + str(min_loss)[0:5]
            log += " - Loss:" + str(epoch_loss)
            if(batch_i == 0):
                out = codecs.open("out.txt", "a","utf-8")
                s = generate_sample(n = 70, init_char = '\n').replace("\n", " ")
                log += " - " + s
                out.write(log)
                out.close()
            if(batch_i % 1 == 0):
                sys.stdout.write(log)
        optim.alpha *= 0.99
        if(save):
            save_ALL()
#-----------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------
#FUNCTION FOR INPUT-TRAIN
def clever_input(s1 = ""):
    s = to_lang(s1)
    if(s[-1] != ' ' and s[-1] != '\n'):
        s += ' '
    #Vocab must consists all vocabulary
    dynamic_indices = np.array(list(map(lambda x:word2index[x], s)))
    
    global batch_size
    batch_size = 1
    n_batches = int((dynamic_indices.shape[0] / (batch_size)))
    
    global bptt
    bptt = n_batches - 1
    
    trimmed_indices = dynamic_indices[:n_batches*batch_size]
    batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()
    
    input_batched_indices = batched_indices[0:-1]
    target_batched_indices = batched_indices[1:]
    
    n_bptt = int(((n_batches-1) / bptt))
    
    global input_batches
    global target_batches
    input_batches = input_batched_indices[:n_bptt*bptt].reshape(n_bptt,bptt,batch_size)
    target_batches = target_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
    
    global min_loss
    min_loss = 1000

    optim.alpha = 0.005
    train(5, False)
    outAnswer_eng = generate_sample(n = 500, init_char = s[-1])
    outAnswer_eng = s + outAnswer_eng

    outAnswer_rus = back_lang(outAnswer_eng)

    return outAnswer_rus
#----------------------------------------------------------------------------------------------------

bot = telebot.TeleBot('YOUR_API')

@bot.message_handler(commands=['start'])

def start_message(message):
    mess = f"<b>Привет, {message.from_user.first_name}</b>!"
    bot.send_message(message.chat.id, mess, parse_mode = 'html')

@bot.message_handler(content_types=['text'])
def send_text(message):

    input_message = message.text
    toOut = clever_input(input_message)
    #toOut is a neural network text
    bot.send_message(message.chat.id, toOut)


bot.polling()
