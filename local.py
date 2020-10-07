import matplotlib.pyplot as plt
import os
import torch as th
import Crypto
from Crypto.PublicKey import RSA
from Crypto import Random

from utils.processors import extend_data


def train_local_model(model_class, word2idx, D, hidden_nodes, data, model_file, 
                batch_size = 256, context_size = 5, epochs = 10, 
                validation_split = 0.2, retrain = False, show_progress = True):
    if os.path.isfile(model_file) and not retrain:
        model = model_class(context_size, len(word2idx), D, word2idx, hidden_nodes)        
        model.load_state_dict(th.load(model_file))        
        for param in model.parameters():
            if param.shape[0] == len(word2idx):
                continue
            param.requires_grad = True
        return model
    random_generator = Random.new().read
    key = RSA.generate(1024, random_generator)
    publickey = key.publickey
    encrypted = publickey.encrypt(model, 32)

    #Request is sent to the backend
    model = model_class(context_size, len(word2idx), D, word2idx, hidden_nodes)
    if os.path.isfile(model_file):                
        model.load_state_dict(th.load(model_file))        
    X, Y = extend_data(data.X, data.Y, context_size)
    X = th.tensor(X)
    Y = th.tensor(Y)
    optimizer = th.optim.RMSprop
    model.fit(X, Y, optimizer, batch_size, epochs, local = True)  
    th.save(model.state_dict(), model_file)
    
    #Response is received and decrypted in order to save
    decrypted = key.decrypt(model)
    return model