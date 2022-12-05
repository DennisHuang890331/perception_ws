import pickle
import numpy as np
import os

def PointcloudSave(data):
    """Testing pointcloud using for costumized Inference API function"""
    tensor = data['points'][0][0].cpu().numpy()
    np.save('pointcloud.npy',tensor)
    print(f'Save {os.getcwd()}/pointcloud.npy success!')

def Pointcloudload(path):
    """Testing pointcloud using for costumized Inference API function"""
    pintcloud = np.load(path)
    print(f'Load {path} success')
    return pintcloud

def SavePrototyeData(data, checkpointName):
    """Save Inference data prototype"""
    prototype = data['img_metas']
    path = checkpointName + '-prototype.p'
    with open(path, 'wb') as file:
        pickle.dump(prototype, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Save {path} sucsess!')    