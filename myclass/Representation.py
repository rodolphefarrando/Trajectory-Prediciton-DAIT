import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib
#matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # Disable StettingWithCopy warning


class Representation:
    """ Representation of trajectories and their predictions"""

    def __init__(self, path, method='',dataset=''):
        """
            :param df data: coordinates, ID and frames.
                   Str method: method to use for prediction
                   string dataser: name of the dataset
                   unique_id: contain all the id in of the file
                   number_traj: gives the number of trajectories in the file
                   dict traj_type: dictionary containing the trajectory type
        """
        self.data = pd.read_csv(path, header = None, names = ['frameNb','id', 'x','y'],delimiter=' ')
        self.method = method
        self.dataset = dataset
        self.unique_id = np.unique(np.array(self.data['id']))
        self.number_traj = len(self.unique_id)
        self.traj_type = {}
        self.traj_type2 = {}


    def representation(self, i):
        """
        :param i:  i is an index from 0 to number_traj and goes into unique_id to find the id
        :return:
        This returns the trajectory of interest shifted and rotate such that the first points is at (0,0)
        and the second one is at (x_1,0).
        This allow a better visualization for the data.

        """

        ## Trajectory of interest, i
        trajectory_i = self.data[self.data['id']==self.unique_id[i]]
        trajectory_i.index = range(len(trajectory_i))
        x_shift, y_shift = trajectory_i['x'][0] ,trajectory_i['y'][0]  ## Shift values
        trajectory_i['x'] = trajectory_i['x']-x_shift # Apply the shift
        trajectory_i['y'] = trajectory_i['y']-y_shift # to all coordinates
        rot_mat=[]

        # This while loop ensure that the pedestrian is moving, otherwise it is not possible to calculate
        # the angle. The angle is calculate for the first pedestrian movement. If the pedestrian is not
        # moving then the corrdinates are (0,0) for all frames
        k = 1
        while sum(trajectory_i.loc[k,['x','y']]==0)==2 and k < len(trajectory_i)-1:
            k += 1
        if k<len(trajectory_i)-1:
            thet = np.arccos(trajectory_i['y'][k]/(np.linalg.norm(trajectory_i.loc[k,['x','y']])))
            if trajectory_i['x'][k]<0:
                thet=-thet
            rot_mat = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]])
            for j in range(len(trajectory_i)):
                trajectory_i.loc[j,['x','y']] = rot_mat.dot(trajectory_i.loc[j,['x','y']]) # Apply the rotation
                                                                                           # to all coordinates

        ##
        id_tmp = self.interaction(i) # Detects the trajectory that interact with trajectory i
        interact = pd.DataFrame()
        if len(id_tmp) > 0:  # If there is interaction apply shift and rotation to the other trajectories
            ind_tmp = self.data['id'].isin(id_tmp)
            interact = self.data.iloc[ind_tmp[ind_tmp > 0].index, :]
            interact['x'] -= x_shift
            interact['y'] -= y_shift
            interact.index = range(len(interact))
            if len(rot_mat) > 0:
                for j in range(len(interact)):
                    interact.loc[j, ['x', 'y']] = rot_mat.dot(interact.loc[j, ['x', 'y']])

        return trajectory_i,interact


    def interaction(self,i):
        """
        :param i:  i is an index from 0 to number_traj and goes into unique_id to find the id
        :return: This function returns the id interacting with trajectory i
        """
        a = self.data[self.data['id']==self.unique_id[i]]
        a.index = range(len(a))

        ## the trajectory having a frame number in common with traj i
        id_tmp = np.unique(np.array(self.data['id'][(self.data['frameNb'] <= max(a['frameNb']))
                                                    & (self.data['frameNb'] >= min(a['frameNb']))]))
        id_tmp = id_tmp[id_tmp != self.unique_id[i]] # remove traj i from the interacting trajectories

        ## This loop ensure that trajectories are sufficiently close to the traj of interest
        ## to actually have an interaction
        for j in id_tmp:
            b = self.data[self.data['id']==j]
            b.index = range(len(b))
            dist = sum(np.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2))/len(a)
            if dist > 5:
                id_tmp = id_tmp[id_tmp != j]

        return id_tmp


    def makePlot(self,i):
        """
        :param i: i is an index from 0 to number_traj and goes into unique_id to find the id
        :return: display a plot with trajectory i and the interacting traj
        """

        ## Some things to chamge here: call representation(i) to have i and interact
        trajectory_i, interact = self.representation(i)
        interact_id = self.interaction(i)

        plt.figure(figsize=(12, 7))
        plt.rc('font', family='serif')
        plt.rc('font', size=20)

        plt.plot(trajectory_i['x'], trajectory_i['y'], marker='+', label='Trajectory of interest')
        if len(interact_id) > 0:
            ind_i = 0
            for w in interact_id:
                ind_i += 1
                plt.plot(interact['x'][interact['id'] == w], interact['y'][interact['id'] == w],
                         color='k', alpha=0.6, marker='.', label='Interact %s'%ind_i)
        plt.legend()
        if max(trajectory_i['y'])>10:
            plt.axis([(-max(trajectory_i['y'])-2)/2, (max(trajectory_i['y'])+2)/2, -1, max(trajectory_i['y'])+1])
        elif len(interact_id) > 0:
            if max(interact['y']) > 10:
                plt.axis([(-max(interact['y']) - 2) / 2, (max(interact['y']) + 2) / 2, -1, max(interact['y']) + 1])
            else:
                plt.axis([-5.5, 5.5, -1, 10])
        else:
            plt.axis([-5.5,5.5, -1,10])
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.savefig(r'../figure/{}_{}.pdf'.format(i, self.dataset), bbox_inches='tight')
        plt.show()


    def makeDynamicPlot(self, i):
        """
        :param i: i is an index from 0 to number_traj and goes into unique_id to find the id
        :return: display a dynamic plot with trajectory i and the interacting traj
        """

        # Previous interaction
        trajectory_i, interact = self.representation(i)
        id_tmp = self.interaction(i)

        ## Initialization of the plot#
        fig = plt.figure(figsize=(12, 7))
        lim_sup = max(max(trajectory_i['y']), max(interact['y'])) + 1
        ax1 = plt.axes(xlim=(-lim_sup/2, lim_sup/2), ylim=(-1, lim_sup))
        line, = ax1.plot([], [], marker='+')

        surround={}
        for w in id_tmp:
            surround[w] = interact[interact['id'] == w]
            surround[w].index = range(len(surround[w]))
            tmp = surround[w][surround[w]['frameNb'] <= trajectory_i.loc[0, 'frameNb']]
            if len(tmp) > 3:
                ax1.plot(tmp.tail(4)['x'], tmp.tail(4)['y'], marker='.', color='k', alpha=0.6)
            else:
                ax1.plot(tmp['x'], tmp['y'], marker='.', color='k', alpha=0.6)

        # Creation of a dictionary with all interacting trajectories
        x_int, y_int = [], []
        dictio = {}
        for j in range(len(id_tmp)):
            dictio['x%s' % j] = []
            dictio['y%s' % j] = []
            dictio['tmp%s' % j] = surround[id_tmp[j]]
            dictio['ind%s' % j] = [k for k,
                            x in enumerate((trajectory_i['frameNb'] == dictio['tmp%s' % j].loc[0, 'frameNb'])) if
                                   x]
            if len(dictio['ind%s' % j]) > 0:
                dictio['ind%s' % j] = dictio['ind%s' % j][0]
            elif dictio['ind%s' % j] == []:
                dictio['ind%s' % j] = - \
                [k for k, x in enumerate((trajectory_i.loc[0, 'frameNb'] == dictio['tmp%s' % j]['frameNb'])) if x][0]

        ## Plot initialization #2
        mar = ['o']
        col = ['k']
        alph = [1, 1, 1]
        lines = []
        for index in range(1+ len(id_tmp)):
            if index < 1:
                lobj = ax1.plot([], [], marker=mar[index], color=col[index], alpha=alph[index])[0]
            else:
                lobj = ax1.plot([], [], marker='.')[0]
            lines.append(lobj)

        # Initialization
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        # Create the animation
        def animate(i):
            label = 'timestep {0}'.format(i)
            if i < len(trajectory_i):
                frame_nb = trajectory_i.loc[i, 'frameNb']
            else:
                frame_nb = trajectory_i['frameNb'].tail(1)
            if i < len(trajectory_i):
                xbis = trajectory_i.loc[i, 'x']
                ybis = trajectory_i.loc[i, 'y']
                x_int.append(xbis)
                y_int.append(ybis)
                for j in range(len(id_tmp)):
                    if (dictio['tmp%s' % j].loc[0, 'frameNb'] <= frame_nb
                        and i - dictio['ind%s' % j] < 20 and dictio['ind%s' % j] != []):
                        dictio['x%s' % j].append(dictio['tmp%s' % j].loc[i - dictio['ind%s' % j], 'x'])
                        dictio['y%s' % j].append(dictio['tmp%s' % j].loc[i - dictio['ind%s' % j], 'y'])

            else:
                for j in range(len(id_tmp)):
                    if i - dictio['ind%s' % j] < 20:
                        dictio['x%s' % j].append(dictio['tmp%s' % j].loc[i - dictio['ind%s' % j], 'x'])
                        dictio['y%s' % j].append(dictio['tmp%s' % j].loc[i - dictio['ind%s' % j], 'y'])

            xlist = [x_int]
            ylist = [y_int]
            for j in range(len(id_tmp)):
                xlist.append(dictio['x%s' % j])
                ylist.append(dictio['y%s' % j])

            ax1.set_xlabel(label)

            for lnum, line in enumerate(lines):
                line.set_data(xlist[lnum], ylist[lnum])  # set data for each line separately.

            return lines

        # Create animation
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=25, interval=500, blit=True)

        # Save animation
        Writer = animation.writers['imagemagick']
        writer = Writer()
        anim.save(r'../figure/dyn_{}_{}.gif'.format(i, self.dataset), writer=writer, dpi=128)


    def totalDistance(self,i):
        """
        :param i: i is an index from 0 to number_traj and goes into unique_id to find the id
        :return: the euclidian distance between first and last point of trajectory i
        """
        trajectory_i, _ = self.representation(i)
        total_distance = np.float(np.sqrt((trajectory_i['x'][0]-trajectory_i['x'].tail(1))**2+
                                          (trajectory_i['y'][0]-trajectory_i['y'].tail(1))**2))

        return total_distance


    def trajectoryType(self,i):
        """
        :param i: i is an index from 0 to number_traj and goes into unique_id to find the id
        :return: A dictionary with the type for each trajectory. The type is defined as below

        6 types of traj:
                1 - ped i = static without interaction
                2 - ped i = static with static interaction
                3 - ped i = static with dynamic interaction
                4 - ped i = dynamic without interaction
                5 - ped i = dynamic with static interaction
                6 - ped i = dynamic with dynamic interaction
                                                               """

        inter = self.interaction(i)         # Is there interaction ?
        total_dist = self.totalDistance(i)  # See if the pedestrian is moving
        indix = []
        for ind in range(len(inter)):
            indix.append(np.where(inter[ind] == self.unique_id)[0][0])
        if total_dist<2:
            if len(inter)==0:
                self.traj_type[i]=1
            else:
                dist =[]
                for j in indix:
                    dist.append(self.totalDistance(j))
                if sum(dist)>2:
                    self.traj_type[i] = 3
                else:
                    self.traj_type[i] = 2

        else:
            if len(inter) == 0:
                self.traj_type[i] = 4
            else:
                dist = []
                for j in indix:
                    dist.append(self.totalDistance(j))
                if sum(dist) > 2:
                    self.traj_type[i] = 6
                else:
                    self.traj_type[i] = 5


    def trajectoryType2(self,i):
        """
        :param i: i is an index from 0 to number_traj and goes into unique_id to find the id
        :return: A dictionary with the type for each trajectory. The type is defined as below

        6 types of traj:
                1 - ped i = Static with or without interaction
                2 - ped i = Dynamic with almost linear traj to predict
                3 - ped i = Dynamic with non linear traj to predict
                                                               """

        total_dist = self.totalDistance(i)  # See if the pedestrian is moving
        if total_dist<2:
            self.traj_type2[i]=1
        else:
            if self.linear(i):
                self.traj_type2[i] = 2
            else:
                self.traj_type2[i] = 3


    def linear(self,i):
        """

        :param i:
        :return: define if the traj to predict is linear
        """

        traj_i, _ = self.representation(i)

        seg1 = np.array(traj_i.loc[10:18,'x':'y'])-np.array(traj_i.loc[9:17,'x':'y'])
        seg2 = np.array(traj_i.loc[11:19,'x':'y'])-np.array(traj_i.loc[10:18,'x':'y'])
        if sum((np.sum(seg1 * seg2, axis=1)/(norm(seg1,axis=1)*norm(seg2,axis=1)))<np.cos(np.pi/9))>0:
            return False

        return True

    def statTrajectoryType(self):
        """

        :return: Return the number of trajectories depending on the type and their index

        """
        count = {}
        index_list = {}
        for i in range(1,7):
            count[i]=0
            index_list[i]=[]

        for key in self.traj_type:
            for i in range(1,7):
                if self.traj_type[key]==i:
                    count[i]+=1
                    index_list[i].append(key)


        return count,index_list

    def statTrajectoryType2(self):
        """

        :return: Return the number of trajectories depending on the type and their index

        """
        count = {}
        index_list = {}
        for i in range(1,4):
            count[i]=0
            index_list[i]=[]

        for key in self.traj_type2:
            for i in range(1,4):
                if self.traj_type2[key]==i:
                    count[i]+=1
                    index_list[i].append(key)


        return count,index_list


    def writeTxt(self):
        """

        :return: write a txt file in the folder with the right type
        #
        """
        if self.dataset=='':
            print('Please enter a number for your dataset (be careful to use unique name to not erase previous files')
        else:
            _, index_list = self.statTrajectoryType()
            for key in index_list:
                index = index_list[key]
                for i in index:
                    traj, interact = self.representation(i)
                    frames = [traj, interact]
                    result = pd.concat(frames)
                    str1 = '%s' % key
                    str2 = '%s' % i
                    np.savetxt(r'../new_data/{}/{}_{}.txt'.format(str1, self.dataset, str2), result.values,
                               fmt=['%d', '%d', '%.8f', '%.8f', '%.8f', '%.8f'])

    def writeTxt2(self):
        """

        :return: write a txt file in the folder with the right type
        #
        """
        if self.dataset=='':
            print('Please enter a number for your dataset (be careful to use unique name to not erase previous files')
        else:
            _, index_list = self.statTrajectoryType2()
            for key in index_list:
                index = index_list[key]
                for i in index:
                    traj, interact = self.representation(i)
                    frames = [traj, interact]
                    result = pd.concat(frames)
                    str1 = '%s' % key
                    str2 = '%s' % i
                    np.savetxt(r'../data_linear/{}/{}_{}.txt'.format(str1, self.dataset, str2), result.values,
                               fmt=['%d', '%d', '%.8f', '%.8f', '%.8f', '%.8f'])

    def speed(self):
        """

        :param i: i is an index from 0 to number_traj and goes into unique_id to find the id
        :return: Return the axis speed of the trajectory i
        """

        self.data['Vx'] = np.zeros(len(self.data))
        self.data['Vy'] = np.zeros(len(self.data))
        for i in range(self.number_traj):
            a,_ = self.representation(i)
            ind = range(20*i,20*(i+1))
            a.index = range(len(a))
            dist1 = a.loc[0:len(a) - 2, 'x':'y']
            dist1.index = range(len(dist1))
            dist2 = a.loc[1:, 'x':'y']
            dist2.index = range(len(dist2))
            dist = dist2 - dist1
            speed_x = np.array(dist['x'] / 0.4)
            speed_y = np.array(dist['y'] / 0.4)
            self.data.loc[ind[1:], 'Vx'] = speed_x
            self.data.loc[ind[1:], 'Vy'] = speed_y


#**************************************************************************
#Linear prediction to put in another class
#    def linearPrediction(self,a):
#        ### Linear prediction
#        mid = np.around(len(a) / 2)
#        a_past = a.loc[0:mid, :]
#        a_pred = a.loc[mid:, :]
#
#        if (a_past['x'][len(a_past) - 1] - a_past['x'][len(a_past) - 2]) == 0:
#            x_lin = a_past['x'][len(a_past) - 1] * np.ones(len(a_pred))
#            diff = a_past['y'][len(a_past) - 1] - a_past['y'][len(a_past) - 2]
#            y_lin = range(len(a_pred)) * diff + a_past['y'][len(a_past) - 1]
#        else:
#            slope = (a_past['y'][len(a_past) - 1] - a_past['y'][len(a_past) - 2]) / (
#            a_past['x'][len(a_past) - 1] - a_past['x'][len(a_past) - 2])
#            ordor = a_past['y'][len(a_past) - 1] - slope * a_past['x'][len(a_past) - 1]
#            diff = a_past['x'][len(a_past) - 1] - a_past['x'][len(a_past) - 2]
#            x_lin = range(len(a_pred)) * diff + a_past['x'][len(a_past) - 1]
#            y_lin = slope * x_lin + ordor
#
#        return x_lin,y_lin
#**************************************************************************

# **************************************************************************
#    def error(self,i):
#        a = self.data[self.data['id']==self.unique_id[i]]
#        a.index = range(len(a))
#        mid = np.around(len(a) / 2)
#        groundTruth = a.loc[mid:, :]
#        if self.methode=='Linear':
#            x_pred, y_pred = self.linearPrediction(a)
#        elif self.methode=='Dcm':
#            x_pred, y_pred = self.linearPrediction(a)
#        else:
#            print('No methods defined ==> No prediciton')
#            x_pred, y_pred = [], []
#            return
#        ## MSE
#        self.av_disp_err = np.append(self.av_disp_err,sum(((groundTruth['y'] - y_pred) ** 2
#                                                        + (groundTruth['x'] - x_pred) ** 2)) / (len(groundTruth)))
#        ## Final displacement
#        self.fin_disp_err = np.append(self.fin_disp_err,sum((groundTruth['y'].tail(1) - y_pred[-1]) ** 2
#                                                        + (groundTruth['x'].tail(1) - x_pred[-1]) ** 2))
#**************************************************************************

# **************************************************************************
### To use in an another class for prediction
#if self.methode=='Linear':
# #    x_pred, y_pred = self.linearPrediction(a)
#elif self.methode=='Dcm':
#    x_pred, y_pred = self.linearPrediction(a)
#else:
#    print('No methods defined ==> No prediciton')
#    x_pred, y_pred = [], []
#**************************************************************************