import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
import matplotlib.pyplot as plt


class Training:

    def __init__(self,path):
        self.data = pd.read_csv(path, header = None, names = ['frameNb','id', 'x','y','Vx','Vy'],delimiter=' ')
        for i in range(len(path) - 1, -1, -1):
            if path[i] == '/':
                self.name = path[i + 1:]
                self.fold = path[i - 1]
                break


    def dataAugmentation(self):
        """

        :return: Create a new trajectory with noise.
        """
        new_data = self.data.copy()
        noise_x = np.random.rand(19) / 10
        noise_y = np.sort(np.random.rand(19) / 10)
        new_data.loc[1:19, 'x'] += noise_x
        new_data.loc[1:19, 'y'] += noise_y
        new_data['Vx'] = np.zeros(len(self.data))
        new_data['Vy'] = np.zeros(len(self.data))
        unique_id = np.unique(np.array(self.data['id']))
        for i in unique_id:
            a = self.data[self.data['id'] == i]
            ind = a.index
            a.index = range(len(a))
            dist1 = a.loc[0:len(a) - 2, 'x':'y']
            dist1.index = range(len(dist1))
            dist2 = a.loc[1:, 'x':'y']
            dist2.index = range(len(dist2))
            dist = dist2 - dist1
            speed_x = np.array(dist['x'] / 0.4)
            speed_y = np.array(dist['y'] / 0.4)
            new_data.loc[ind[1:], 'Vx'] = speed_x
            new_data.loc[ind[1:], 'Vy'] = speed_y
        name = 'aug_'+self.name
        np.savetxt(r'../new_data/{}/{}'.format(self.fold,name),new_data.values,fmt=['%d', '%d', '%.8f', '%.8f', '%.8f', '%.8f'])

    def flip(self):
        """

        :return: flip the traj
        """
        new_data = self.data.copy()
        new_data['x']=-new_data['x']
        new_data['Vx'] = -new_data['Vx']
        name = 'flip'+self.name
        np.savetxt(r'../new_data/{}/{}'.format(self.fold, name), new_data.values,
                   fmt=['%d', '%d', '%.8f', '%.8f', '%.8f', '%.8f'])

    def traj_plot(self):


        plt.figure(figsize=(12, 7))
        plt.rc('font', family='serif')
        plt.rc('font', size=20)

        nb = np.int(len(self.data)/20)

        for i in range(nb):
            plt.plot(self.data.loc[20*i:20*(i+1)-1,'x'], self.data.loc[20*i:20*(i+1)-1,'y'])

        plt.axis([-5.5, 5.5, -1, 10])
        #plt.savefig(r'../figure/{}.pdf'.format(self.name), bbox_inches='tight')