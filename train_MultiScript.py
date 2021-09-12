import numpy as np
from models.Train_BigGANV1 import train
#from models.Train_fn_V3 import train

if __name__=="__main__":
    batch_sizes = [128]
    channels = [32]

    data_name = "C100"
    image_size = 32
    use_SN = True
    dataroot='./data/'
    #model_path = './ImagesForScores/EnsembleV8+Encoder/C10/32/32x32/28000.npz'
    #model_path = './log/DisV3+gamma1+attention+kernel=31/celeba/128/32x8/Epoch_20000_checkpoint.pth'
    model_path = ''
    for batch_size in batch_sizes:
        for ch in channels:
            train(data_name, channel=ch, batch_size=batch_size, image_size=image_size, 
                                            use_SN=use_SN, epoch_of_updateG=1, dataroot=dataroot, cuda=True, model_path=model_path)
        
