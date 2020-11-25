# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:59:49 2020

@author: Lyy
"""

from gurobipy import *
import numpy as np
import re
import pandas as pd
import random
from pyproj import Transformer
import datetime

class Node(object):
    idcase = {}
    def __init__(self, nid, ntype, x, y):
        self.id = nid
        self.type = ntype
        self.x = x
        self.y = y
        self.link_approach = []
        self.link_exit = []
        Node.idcase[self.id] = self
    
class Link(object):
    idcase = {}
    def __init__(self, lid, from_nd, to_nd, vf, num_of_lanes, exit_nols, w=12, kj=120, qmax=1800):
        self.id = lid
        self.from_node = from_nd
        self.to_node = to_nd
        self.num_of_lanes = num_of_lanes
        self.vf = vf
        self.w = w
        self.kj = kj
        self.qmax = qmax
        self.flowinput = []
        self.exit_nols = exit_nols
        Link.idcase[self.id] = self
        
class Movement(object):
    idcase = {}
    def __init__(self, gdir, from_link, to_link):
        self.gdirect = gdir # 1 = Left, 2 = Stright, 3 = Right
        self.from_link = from_link
        self.to_link = to_link
        self.move = str(from_link) + ',' + str(to_link)
        self.ratio = 0
        Movement.idcase[str(from_link) + ',' + str(to_link)] = self
        
    def getAllMovement():
        df = pd.DataFrame(columns=["Movement", "gdirect", "Corr_Node"])
        index = 0
        temp_df = pd.DataFrame(columns=["Movement", "gdirect", "Corr_Node"])
        from_link = 0
        for key in Movement.idcase:
            if from_link == 0 or from_link == Movement.idcase[key].from_link:
                temp_df.loc[index] = [key, 
                                      Movement.idcase[key].gdirect, 
                                      Node.idcase[Link.idcase[Movement.idcase[key].from_link].to_node].id]
                
                from_link = Movement.idcase[key].from_link
            else:
                temp_df = temp_df.sort_values(by="gdirect")
                df = df.append(temp_df)
                temp_df = pd.DataFrame(columns=["Movement", "gdirect", "Corr_Node"])
                temp_df.loc[index] = [key, 
                                      Movement.idcase[key].gdirect, 
                                      Node.idcase[Link.idcase[Movement.idcase[key].from_link].to_node].id]
                from_link = 0
                
            index += 1
            
        temp_df = temp_df.sort_values(by="gdirect")
        df = df.append(temp_df)       
        df.to_csv("movement.csv")
            
    def getMovementByFromLink(linkid):
        tempList = []
        for key in Movement.idcase:
            if linkid == Movement.idcase[key].from_link:
                tempList.append(Movement.idcase[key])
            else:
                continue
            
        return tempList
        
def getEuclideanDis(x1, x2, y1, y2):
    return np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))

def getManhattanDis(x1, x2, y1, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)

def getLength(x,y):
    return np.sqrt(x**2 + y**2)

def getAngle(x1, x2, y1, y2):
    return np.degrees(np.arccos(np.abs(x2*x1 + y2*y1) / getLength(x1,y1) / getLength(x2,y2)))

def getCosine(x1, x2, y1, y2):
    return (x2*x1 + y2*y1) / getLength(x1,y1) / getLength(x2,y2)

def getCrossProduct(va, vb):
    return va[0]*vb[1] - va[1]*vb[0]

class pop(object):
    def __init__(self, chromo):
        self.fitness = 0
        self.chromo = chromo 
        
    def mutation(self):
        if len(self.chromo) == 0:
            print("Bad instance")
        else:
            cycle_list = [100, 110, 120, 130, 140, 150, 160, 170, 180]
            pos_of_mut = random.randint(0, len(self.chromo) - 1)
            self.chromo[pos_of_mut] = random.choice(cycle_list)
                
    def newCrossover(self, couple):
        tempList = []
        for i in range(len(self.chromo)):
            if self.chromo[i] != couple.chromo[i]:
                tempList.append(i)
            
        pos_of_crs = random.choice(tempList)
            
        temp_chromo = self.chromo[pos_of_crs]
        self.chromo[pos_of_crs] = couple.chromo[pos_of_crs]
        couple.chromo[pos_of_crs] = temp_chromo
            
    def crossover(self, couple):
        pos_of_crs = random.randint(0, len(self.chromo))
        new_chromo = []
        new_chromo.extend(self.chromo[0: pos_of_crs])
        if pos_of_crs == len(self.chromo):
            pos_of_crs = pos_of_crs - 1
            
        new_chromo.append(couple.chromo[pos_of_crs])
        new_chromo.extend(self.chromo[pos_of_crs + 1: len(couple.chromo)])
        
        new_pop = pop(new_chromo)
        return new_pop

    def get_fitness(self):
        fitness = lp_optimize(1, 1, input_chromo=self.chromo, isReturn=True)
        while fitness == 0:
            self.mutation()
            fitness = lp_optimize(1, 1, input_chromo=self.chromo, isReturn=True)
        
        lp_optimize(1, 1, input_chromo=self.chromo, isReturn=True)
        # fitness += getSD()
        
        if fitness != 0:
            self.fitness = fitness
        else:
            self.fitness = 1000000

def getSum(fitlist):
    total = 0
    for key in fitlist:
        total += fitlist[key]
    
    return total

def getSD():
    df = pd.read_csv("output_density.csv")
    # df = df.drop(9, axis=0, inplace=False)
    std = 0
    for col in df:
        std += df[col].std()
        
    return std

def getCumulativeSum(fitlist):
    # total = getSum(fitlist)
    for key in fitlist:
        fitlist[key] = max(fitlist.values()) - fitlist[key] + 1
        
    for key in fitlist:
        if fitlist[key] == 0:
            continue
        fitlist[key] = np.exp(np.log(fitlist[key]) / np.log(2))
    
    total = getSum(fitlist)
    for key in fitlist:
        fitlist[key] = fitlist[key] / total
    return fitlist

def getNewFitnessList(poplist):
    # Since the lower delay of a intersection is better, we need to transform origin fitness list.
    # New fitness list is consisted of: sum of origin list - fitness value 
    newfl = {}
    fitness_list = []
    for elem in poplist:
        fitness_list.append(elem.fitness)
    
    sum_of_fitness = sum(fitness_list)
        
    for elem in poplist:
        newfl[elem] = elem.fitness
        
    return newfl

def selection(poplist):
    cum_probability_list = getCumulativeSum(getNewFitnessList(poplist))
    subtra = 0
    for key in cum_probability_list:
        if random.uniform(0, 1 - subtra) < cum_probability_list[key]:
            return key
        else:
            subtra += cum_probability_list[key]
        
def readNetwork():
    link_file = pd.read_csv("link.csv")
    node_file = pd.read_csv("node.csv")
    
    for i in range(len(link_file)):
        Link(*list(link_file.iloc[i]))
    
    for j in range(len(node_file)):
        Node(*list(node_file.iloc[j]))
        
    for key in Node.idcase:
        if Node.idcase[key].type == 0:
            for subkey in Link.idcase:
                if Link.idcase[subkey].from_node == Node.idcase[key].id:
                    Node.idcase[key].link_exit.append(Link.idcase[subkey])
                elif Link.idcase[subkey].to_node == Node.idcase[key].id:
                    Node.idcase[key].link_approach.append(Link.idcase[subkey])
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        new_coord = {'X':0, 'Y':0}
        new_coord['X'],new_coord['Y'] = transformer.transform(Node.idcase[key].x, Node.idcase[key].y)
        Node.idcase[key].x = new_coord['X']
        Node.idcase[key].y = new_coord['Y']
                    
def initMovement():
    for key in Node.idcase:
        for app_link in Node.idcase[key].link_approach:
            app_vector = [Node.idcase[app_link.to_node].x - Node.idcase[app_link.from_node].x, 
                              Node.idcase[app_link.to_node].y - Node.idcase[app_link.from_node].y]
            for exit_link in Node.idcase[key].link_exit:
                exit_vector = [Node.idcase[exit_link.to_node].x - Node.idcase[exit_link.from_node].x, 
                              Node.idcase[exit_link.to_node].y - Node.idcase[exit_link.from_node].y]
                
                cosine = getCosine(app_vector[0], exit_vector[0], app_vector[1], exit_vector[1])
                if getCrossProduct(app_vector, exit_vector) > 0:
                    right_flag = 0
                else:
                    right_flag = 1
                
                if -0.707 < cosine <= 0.707 and right_flag:
                    Movement(3, app_link.id, exit_link.id)
                elif 0.707 < cosine <= 1:
                    Movement(2, app_link.id, exit_link.id)
                elif -0.707 < cosine <= 0.707 and not right_flag:
                    Movement(1, app_link.id, exit_link.id)
        
def initConflict():
    conflict = []
    for key in Movement.idcase:
        if Movement.idcase[key].gdirect == 3:
            continue
        from_link = Link.idcase[Movement.idcase[key].from_link]
        to_link = Link.idcase[Movement.idcase[key].to_link]
        fl_vector = [Node.idcase[from_link.to_node].x - Node.idcase[from_link.from_node].x, 
                     Node.idcase[from_link.to_node].y - Node.idcase[from_link.from_node].y]
        
        other_app_link = Node.idcase[from_link.to_node].link_approach.copy()
        other_app_link.remove(from_link)
        
        for other_link in other_app_link:
            ol_vector = [Node.idcase[other_link.to_node].x - Node.idcase[other_link.from_node].x, 
                         Node.idcase[other_link.to_node].y - Node.idcase[other_link.from_node].y]
            
            movelist = Movement.getMovementByFromLink(other_link.id)
            cosine = getCosine(fl_vector[0], ol_vector[0], fl_vector[1], ol_vector[1])
            if -0.707 < cosine <= 0.707:
                for elem in movelist:
                    if elem.gdirect == 3:
                        continue
                    
                    other_to_link = Link.idcase[elem.to_link]
                    # if to_link == other_to_link:
                    #     continue
                    
                    conflict.append([from_link.id, to_link.id, elem.from_link, elem.to_link])
            elif 0.707 < cosine <= 1 or cosine < -0.707:
                if Movement.idcase[key].gdirect == 2:
                    for elem in movelist:
                        if elem.gdirect == 1:
                            conflict.append([from_link.id, to_link.id, elem.from_link, elem.to_link])
                else:
                    for elem in movelist:
                        if elem.gdirect == 2:
                            conflict.append([from_link.id, to_link.id, elem.from_link, elem.to_link])
                            
    return conflict

def getOtherApproachLink(link):
    other_app_link = Node.idcase[link.to_node].link_approach.copy()
    other_app_link.remove(link)
    return other_app_link

def getOtherExitLink(link):
    other_exit_link = Node.idcase[link.to_node].link_exit.copy()
    other_exit_link.remove(link)
    return other_exit_link

def getExitLinkByApproachLink(n):
    tempList = []
    for key in Movement.idcase:
        # if Movement.idcase[key].gdirect == 3:
        #     continue
        if Movement.idcase[key].from_link == n:
            tempList.append(Movement.idcase[key].to_link)
            
    return tempList

def getApproachLinkByExitLink(n):
    tempList = []
    for key in Movement.idcase:
        # if Movement.idcase[key].gdirect == 3:
        #     continue
        if Movement.idcase[key].to_link == n:
            tempList.append(Movement.idcase[key].from_link)
            
    return tempList

def calibrateFlowInput():
    cali_file = pd.read_csv("flow_input_larger.csv")
    for key in Link.idcase:
        Link.idcase[key].flowinput = cali_file.where(cali_file['id']==key).dropna().iloc[0]['flow_input']
        
def calibrateRatio():
    ratio_file = pd.read_csv("ratio.csv")
    for key in Movement.idcase:
        Movement.idcase[key].ratio = ratio_file.where(ratio_file['id']==key).dropna().iloc[0]['ratio']
        
def calibrate_all():
    cf = pd.read_csv("movement_calibration.csv")
    flow_input = {Link.idcase[key].id:[] for key in Link.idcase}
    for key in Movement.idcase:
        Movement.idcase[key].ratio = cf.where(cf['Movement'] == key).dropna().iloc[0]['Ratio']
        
        linkID = Movement.idcase[key].from_link
        flow_input[linkID].append(cf.where(cf['Movement']==key).dropna().iloc[0]['Flow'])
        
    for key in flow_input:
        if flow_input[key]:
            Link.idcase[key].flowinput = np.random.poisson(lam=sum(flow_input[key]), size=180) / 3600 * 5
            
    Link.idcase[1].flowinput = [0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Link.idcase[7].flowinput = [0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 7.999999999994523, 7.999999999994523, 7.999999999994523, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.1650393417237447e-09, 3.1650393417237447e-09, 3.1650393417237447e-09, 0, 0, 0]
    
    for i in range(len(Link.idcase[1].flowinput)):
        Link.idcase[1].flowinput[i] = Link.idcase[1].flowinput[i] / 7.5
        Link.idcase[7].flowinput[i] = Link.idcase[7].flowinput[i] / 7.5

def lp_optimize(c, init_theta={}, init_phi={}, init_theta_upper={}, init_phi_upper={}, input_chromo=[], isReturn=False):
    try:
        m = Model('si%i'%(c))
        if len(input_chromo):
            cycle_dict = {3:input_chromo[0], 5:input_chromo[1], 10:input_chromo[2], 11:input_chromo[3]}
        else:
            cycle_dict = {3:100, 5:160, 10:100, 11:100}
        clock = 5
        gap = clock #clearance time
        gap = 0
        mingreen = 10
        Totaltime = 200
        maxcell = 3
        Tc = int(Totaltime/clock)
        cc = int(c/clock)
        
        cc_dict = {}
        for key in cycle_dict:
            cc_dict[key] = int(cycle_dict[key] / clock)
            
        w = 20  #shock wave speed
        v = 60   #free flow speed
        vf = 60
        d = 8    #jam headway in meters
        K = {}
        U = []
        S = []
        E = []
        # 需要注意的是，S和E里存储的是Link类的实例，而U里存储的是id
        # Note that instances of Link class are stored in S and E while id of links are stored in U. 
        for key in Node.idcase:
            S.extend(Node.idcase[key].link_approach)
            E.extend(Node.idcase[key].link_exit)
        
        for key in Link.idcase:
            if Link.idcase[key] in S:
                K[Link.idcase[key].id] = Link.idcase[key].num_of_lanes
            elif Link.idcase[key] in E:
                K[Link.idcase[key].id] = Link.idcase[key].exit_nols
                    
            U.append(Link.idcase[key].id)
            
        # 交叉口进口道的车道数，可以与路段车道数区分开来设置，这里并未使用。
        # Number of lanes of apporach link of intersections, different with number of lanes of link. 
        K_app = {}
        K_app[1] = 3
        K_app[14] = 6
        K_app[15] = 6
        K_app[4] = 6
        K_app[17] = 6
        K_app[16] = 3
        K_app[22] = 5
        K_app[21] = 3
        K_app[18] = 3
        K_app[19] = 5
        K_app[10] = 5
        K_app[20] = 5
        K_app[7] = 3

        s = 2105     #saturated flow rate for one lane
        Kjam = {}
        PQ = {}
                
        for i in U:
            PQ[i] = [(s/3600*clock)*K[i] for t in range(Tc+1)]
            Kjam[i] = (v/3.6*clock*1/d)*K[i]
    
        M = 100000 #infinite number
        
        Demand = 1800
        initial = 0
        #gdirect = 1 signs left-turn, and 2 right-turn,
        #arm signs intersection arm
        movement_dict = {}
        for key in Movement.idcase:
            movement_dict[(Movement.idcase[key].from_link, Movement.idcase[key].to_link)] \
                        = [Movement.idcase[key].ratio, Movement.idcase[key].gdirect]
                
        movement, ratio, gdirect = multidict(movement_dict)
        
        conflict_list = initConflict()
        conflict_dict = {}
        for elem in conflict_list:
            conflict_dict[tuple(elem)] = 1
            
        conflict, value = multidict(conflict_dict)
        
        sta = {}
        phi = {}
        zs = {}
        STA={}
        PHI={}
        
        # 如果没有给定初始解，则直接求解，否则读取初始解。
        # Read initial solution if it exists or solve the model directily.
        if len(init_theta) == 0 and len(init_phi) == 0:
            for i,i1 in movement:
                nd = Link.idcase[i].to_node
                sta[i,i1] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='sta%s_%s_'%(i,i1))
                phi[i,i1] = m.addVar(lb = mingreen / cycle_dict[nd],ub=1.0,vtype=GRB.CONTINUOUS, name='phi%s_%s'%(i,i1)) 
                zs[i,i1] = m.addVar(vtype=GRB.BINARY, name='zs%s_%s'%(i,i1))
            
            for elem in S:
                i = elem.id
                nd = Link.idcase[i].to_node
                for k in range(1,K[i]+1):
                    STA[i,k] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='STA%s_%s'%(i,k))
                    PHI[i,k] = m.addVar(lb = mingreen / cycle_dict[nd], ub=1.0,vtype=GRB.CONTINUOUS, name='PHI%s_%s'%(i,k))
        else:
            # 可以选择是否直接使用单交叉口的结果而仅调节相序，但是目前来看，效果并不好，所以注释掉了。
            # One may uncomment line 480 and line 490 to fix timing of all phases and adjuct offset only.
            for i,i1 in movement:
                nd = Link.idcase[i].to_node
                sta[i,i1] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='sta%s_%s_'%(i,i1))
                phi[i,i1] = m.addVar(lb = mingreen / cycle_dict[nd],ub=1.0,vtype=GRB.CONTINUOUS, name='phi%s_%s'%(i,i1)) 
                zs[i,i1] = m.addVar(vtype=GRB.BINARY, name='zs%s_%s'%(i,i1))
                sta[i,i1].start = init_theta[i,i1]
                phi[i,i1].start = init_phi[i,i1]
                # m.addConstr(sta[i,i1] == init_theta[i,i1])
#                m.addConstr(phi[i,i1] == init_phi[i,i1])
            for elem in S:
                i = elem.id
                nd = Link.idcase[i].to_node
                for k in range(1,K[i]+1):
                    STA[i,k] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='STA%s_%s'%(i,k))
                    PHI[i,k] = m.addVar(lb = mingreen / cycle_dict[nd], ub=1.0,vtype=GRB.CONTINUOUS, name='PHI%s_%s'%(i,k))
                    STA[i,k].start = init_theta_upper[i,k]
                    PHI[i,k].start = init_phi_upper[i,k]
                    # m.addConstr(STA[i,k] == init_theta_upper[i,k])
#                    m.addConstr(PHI[i,k] == init_phi_upper[i,k])
                
        omg={}
        for i,i1,i2,i3 in conflict:
                omg[i,i1,i2,i3]= m.addVar(vtype=GRB.BINARY, name='omg%s_%s_%s_%s'%(i,i1,i2,i3)) 
        
        volve = {}
        fI = {}
        for i,i1 in movement:
            for t in range(1, Tc+1):
                fI[i,i1,t] = m.addVar(lb=0.0,vtype=GRB.CONTINUOUS, name='fI%s_%s_%s'%(i,i1,t))
                volve[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='volve%s_%s_%s'%(i,i1,t))
                if Movement.idcase[str(i)+','+str(i1)].gdirect == 3:
                    m.addConstr(volve[i,i1,t] == 1)
        zs1={}
        zs2={}
        z3={}
        z4={}
        # zn = {}
    #zs1 and zs2: integer variable for signal settings
    #z3 and z4: integer variable for movement volve
        for i,i1 in movement:
            for t in range(1, Tc+1):
                zs1[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs1%s_%s_%s'%(i,i1,t))
                zs2[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs2%s_%s_%s'%(i,i1,t))
                z3[i,i1,t]= m.addVar( vtype=GRB.BINARY, name='z3%s_%s_%s'%(i,i1,t))
                z4[i,i1,t]= m.addVar( vtype=GRB.BINARY, name='z4%s_%s_%s'%(i,i1,t))
        z1={}
        z2={}
        n={}
                
    # z1 and z2: integer variable for flow constrains
    # n: accumulation
        for i in U:
            for j in range(1,maxcell):
                for t in range(1, Tc+2):
                    n[i,j,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='n%s_%s_%s'%(i,j,t))
                    z1[i,j,t]= m.addVar( vtype=GRB.BINARY, name='z1%s_%s_%s'%(i,j,t))
                    z2[i,j,t]= m.addVar( vtype=GRB.BINARY, name='z2%s_%s_%s'%(i,j,t))
            
        f={}
    #f: flow
        for i in U:
            for j in range(1,maxcell+1):
                for t in range(1, Tc+1):
                    f[i,j,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='f%s_%s_%s'%(i,j,t))
                
        # initial accumulation is 0
        for i in U:
            for j in range(1,maxcell):
                m.addConstr(n[i,j,1] == initial)
   
    # LTM
        N = {}
        G = {}
        Nr = {}
        Sending = {}
        Receiving = {}
        zx1 = {}
        zx2 = {}
        zx3 = {}
        zx4 = {}
        zx5 = {}
        freeflow = {}
                
        for i in U:
            for t in range(Tc+1):
                for j in [0,1]:
                    N[i,j,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='N%s_%s_%s'%(i,j,t))
                G[i,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='G%s_%s'%(i,t))
                Receiving[i,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='R%s_%s'%(i,t))
                Sending[i,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='S%s_%s'%(i,t))
                zx1[i,t] = m.addVar( vtype=GRB.BINARY, name='z1%s_%s'%(i,t))
                zx2[i,t] = m.addVar( vtype=GRB.BINARY, name='z2%s_%s'%(i,t))
                zx3[i,t] = m.addVar( vtype=GRB.BINARY, name='z3%s_%s'%(i,t))
                zx4[i,t] = m.addVar( vtype=GRB.BINARY, name='z4%s_%s'%(i,t))
                zx5[i,t] = m.addVar( vtype=GRB.BINARY, name='z5%s_%s'%(i,t))
                freeflow[i,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='ff%s_%s'%(i,t))
                        
        for i in U:
            for t in range(1, Tc+1):
                linkLength = getEuclideanDis(Node.idcase[Link.idcase[i].from_node].x, 
                                             Node.idcase[Link.idcase[i].to_node].x, 
                                             Node.idcase[Link.idcase[i].from_node].y, 
                                             Node.idcase[Link.idcase[i].to_node].y) / 1000
                        
                bwtt = t - int((linkLength / w) / clock * 3600)
                fftt = t - int((linkLength / vf) / clock * 3600)
                
                # 定义路段上的接收能力
                # Define Receiving capacity
                if bwtt <= 0:
                    m.addConstr(Receiving[i,t-1] <= 0 + 120*K[i] * linkLength - N[i,0,t-1])
                    m.addConstr(Receiving[i,t-1] >= 0 + 120*K[i] * linkLength - N[i,0,t-1] - M * zx1[i,t])
                else:
                    m.addConstr(Receiving[i,t-1] <= N[i,1,bwtt] + 120*K[i] * linkLength - N[i,0,t-1])
                    m.addConstr(Receiving[i,t-1] >= N[i,1,bwtt] + 120*K[i] * linkLength - N[i,0,t-1] - M * zx1[i,t])
                            
                m.addConstr(Receiving[i,t-1] <= PQ[i][t])
                m.addConstr(Receiving[i,t-1] >= PQ[i][t] - M * (1 - zx1[i,t]))
                
                # 定义传输流，判断上游是否来自一个起点（6代表起点而0代表交叉口）
                # Define transition flow, where node type 6 represents dead_end (sink or source) and 0 represents intersection.
                if Node.idcase[Link.idcase[i].from_node].type == 6:
                    elem = Link.idcase[i]
                    m.addConstr(G[i,t-1] <= gurobipy.quicksum(elem.flowinput[tau] for tau in range(1, t+1)) - N[i,0,t-1])
                    m.addConstr(G[i,t-1] >= gurobipy.quicksum(elem.flowinput[tau] for tau in range(1, t+1)) - N[i,0,t-1] - M * zx2[i,t])
                else:
                    m.addConstr(G[i,t-1] <= quicksum(quicksum(fI[j,i,t1] for j in getApproachLinkByExitLink(i))
                                                     for t1 in range(1, t)) - N[i,0,t-1])
                    m.addConstr(G[i,t-1] >= quicksum(quicksum(fI[j,i,t1] for j in getApproachLinkByExitLink(i)) 
                                                     for t1 in range(1, t)) - N[i,0,t-1] - M * zx2[i,t])
                    
                m.addConstr(G[i,t-1] <= Receiving[i,t-1])
                m.addConstr(G[i,t-1] >= Receiving[i,t-1] - M * (1 - zx2[i,t]))
                        
                m.addConstr(N[i,0,t] == N[i,0,t-1] + G[i,t-1])

                # 定义路段上的传输能力，向下游传输的能力要考虑Cell内的剩余空间
                # Define sending capacity. Spare space of downstream cells are considered.
                if Node.idcase[elem.to_node].type == 0:
                    if fftt <= 0:
                        m.addConstr(Sending[i,t-1] <= 0 - N[i,1,t-1])
                        m.addConstr(Sending[i,t-1] >= 0 - N[i,1,t-1] - M * zx3[i,t])
                    else:
                        m.addConstr(Sending[i,t-1] <= N[i,0,fftt] - N[i,1,t-1])
                        m.addConstr(Sending[i,t-1] >= N[i,0,fftt] - N[i,1,t-1] - M *zx3[i,t])
                                
                    m.addConstr(Sending[i,t-1] <= PQ[i][t])
                    m.addConstr(Sending[i,t-1] >= PQ[i][t] - M * zx4[i,t])
                    m.addConstr(Sending[i,t-1] <= w/v * (Kjam[i] - n[i,1,t]))
                    m.addConstr(Sending[i,t-1] >= w/v * (Kjam[i] - n[i,1,t]) - M * (2-zx3[i,t]-zx4[i,t]))
                else:
                    if fftt <= 0:
                        m.addConstr(Sending[i,t-1] <= 0 - N[i,1,t-1])
                        m.addConstr(Sending[i,t-1] >= 0 - N[i,1,t-1] - M * zx3[i,t])
                    else:
                        m.addConstr(Sending[i,t-1] <= N[i,0,fftt] - N[i,1,t-1])
                        m.addConstr(Sending[i,t-1] >= N[i,0,fftt] - N[i,1,t-1] - M *zx3[i,t])
                                
                    m.addConstr(Sending[i,t-1] <= PQ[i][t])
                    m.addConstr(Sending[i,t-1] >= PQ[i][t] - M * (1 - zx3[i,t]))
                
                m.addConstr(N[i,1,t] == N[i,1,t-1] + Sending[i,t-1])
                
                # 第一个Cell的输入流量，即为路段的传输能力大小
                # Flow into the first cell of intersections equal to sending capacity of the link.
                elem = Link.idcase[i]
                if Node.idcase[elem.to_node].type == 0:
                    m.addConstr(f[i,1,t] == Sending[i,t])
                        
    #CTM update
        for i in U:
            for j in range(1,maxcell):
                for t in range(1,Tc+1):
                    m.addConstr(n[i,j,t+1] == n[i,j,t] + f[i,j,t] - f[i,j+1,t])
                
        # Coordination
        for elem in E:
            i1 = elem.id
            for t in range(1, Tc+1):
                m.addConstr(Receiving[i1,t] >= quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))    
                        
        #last cell flow on link i in S
        for elem in S:
            i = elem.id
            for t in range(1, Tc+1):
                m.addConstr(f[i,maxcell,t] == quicksum(fI[i,i1,t] for i1 in getExitLinkByApproachLink(i)))      
        
                
        # Lock Right turn movement with all green
        for key in movement_dict:
            if movement_dict[key][1] == 3:
                m.addConstr(sta[key[0], key[1]] == 0)
                m.addConstr(phi[key[0], key[1]] == 1)
                
        DeRT={}
        for i,i1 in movement:
            for k in range(1,K[i]+1):
                DeRT[i,i1,k]=m.addVar(vtype=GRB.BINARY,name='DeRT%s_%s_%s'%(i,i1,k))

    #1 lane permission
    #2. one lane can share multiple directions
    # constrain 28
        for elem in S:
            i = elem.id
            for k in range(1,K[i]+1):
                m.addConstr(quicksum(DeRT[i,i1,k] for i1 in getExitLinkByApproachLink(i))>=1)
                
    #share less than two directions
    #3.one movement can distribute on multiple lanes
    # constrain 29
        for i,i1 in movement:
            m.addConstr(quicksum(DeRT[i,i1,k] for k in range(1,K[i]+1))>=1)
    
    # #4.flow distribution
    # #5. no conflict for adjacent lane settings
        #here 'k+1' constrains that the number of lanes must bigger than one
        # constrain 30
        for elem in S:
            i = elem.id
            for i1 in getExitLinkByApproachLink(i):
                for i2 in getExitLinkByApproachLink(i):
                    for k in range(1,K[i]):
                        if (gdirect[i,i1] > gdirect[i,i2]):
                            m.addConstr(DeRT[i,i1,k+1]+DeRT[i,i2,k]<=1)
        
        # 对于转向比更大的movement，保证车道数不少于转向比更小的movement
        # For a movement has larger turning ratio than the other movement, assign lane to the movement not less than the other movement.                   
        for elem in S:
            i = elem.id
            for i1 in getExitLinkByApproachLink(i):
                for i2 in getExitLinkByApproachLink(i):
                    if(ratio[i,i1]>ratio[i,i2]):
                        m.addConstr(quicksum(DeRT[i,i1,k] for k in range(1,K[i]+1))>=quicksum(DeRT[i,i2,k] for k in range(1,K[i]+1)))
                
    # constrain 37 and 38
        for elem in S:
            i = elem.id
            for i1 in getExitLinkByApproachLink(i):
                for k in range(1,K[i]+1):
                    m.addConstr(M*(1-DeRT[i,i1,k]) >= STA[i,k]-sta[i,i1])
                    m.addConstr(-M*(1-DeRT[i,i1,k]) <= STA[i,k]-sta[i,i1])
                    m.addConstr(M*(1-DeRT[i,i1,k]) >= PHI[i,k]-phi[i,i1])
                    m.addConstr(-M*(1-DeRT[i,i1,k]) <= PHI[i,k]-phi[i,i1])
                
        for i,i1 in movement:
            m.addConstr(sta[i,i1]+phi[i,i1]<=1)
            nd = Link.idcase[i].to_node
            for t in range(1, cc_dict[nd]+1):
            # for t in range(1, cc+1):
                # constrain 14
                m.addConstr(t >= cycle_dict[nd]/clock * sta[i,i1] - (1 - volve[i,i1,t]) * M)
                m.addConstr(t <= cycle_dict[nd]/clock * (sta[i,i1] + phi[i,i1]) + (1 - volve[i,i1,t])*M)
                
                # constrain 15
                m.addConstr(t >= cycle_dict[nd]/clock * (sta[i,i1] + phi[i,i1]) - volve[i,i1,t]*M- (1-zs1[i,i1,t]) * M)
                m.addConstr(t <= cycle_dict[nd]/clock * sta[i,i1] + volve[i,i1,t]* M + zs1[i,i1,t] * M)
                
    #order of movement   
        for i,i1,i2,i3 in conflict:
            m.addConstr(omg[i,i1,i2,i3]+omg[i2,i3,i,i1]==1)
            m.addConstr(sta[i,i1]+phi[i,i1]+gap/c <= sta[i2,i3]+omg[i,i1,i2,i3])

    #make sure that the value of variables volve is the same in every cycle
        for i,i1 in movement:
            nd = Link.idcase[i].to_node
            cc = cc_dict[nd]
            for t in range(1,Tc+1-cc):
                m.addConstr(volve[i,i1,t]==volve[i,i1,t+cc])
                
        lane_occu = {}
        for elem in S:
            i = elem.id
            for k in range(1, 4):
                for t in range(Tc+2):
                    lane_occu[i,k,t] = m.addVar(lb = 0, vtype=GRB.CONTINUOUS, name='lb%s_%s_%i'%(i,k,t))
                            
        f_star = {}
        for elem in S:
            i = elem.id
            exits = getExitLinkByApproachLink(i)
            for i1 in exits:
                for t in range(Tc+2):
                    f_star[i,i1,t] = m.addVar(lb = 0, vtype=GRB.CONTINUOUS, name='f_star%s_%s_%i' % (i,i1,t))
                            
        zs3={}
        zs4={}

        for i,i1 in movement:
            for t in range(1, Tc+1):
                zs3[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs3%s_%s_%s'%(i,i1,t))
                zs4[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs4%s_%s_%s'%(i,i1,t))
        
        # f_star is the flow into each lane correspond to each movements.                
        for elem in S:
            i = elem.id
            exits = getExitLinkByApproachLink(i)
            for i1 in exits:
                move = Movement.idcase[str(i)+','+str(i1)]
                for t in range(1, Tc+1):
                    m.addConstr(f_star[i,i1,t] <= n[i,maxcell-2,t] * ratio[i,i1])
                    m.addConstr(f_star[i,i1,t] >= n[i,maxcell-2,t] * ratio[i,i1]-zs3[i,i1,t]*M)
                                
                    m.addConstr(f_star[i,i1,t] <= PQ[i][t] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i])
                    m.addConstr(f_star[i,i1,t] <= w/v*(Kjam[i] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i] - lane_occu[i,move.gdirect,t]))
                                
                    m.addConstr(f_star[i,i1,t] >= PQ[i][t] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i] - zs4[i,i1,t]*M)
                    m.addConstr(f_star[i,i1,t] >= w/v*(Kjam[i] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i] - lane_occu[i,move.gdirect,t]) - (2-zs3[i,i1,t]-zs4[i,i1,t])*M)
                            
        for i,i1 in movement:
            move = Movement.idcase[str(i)+','+str(i1)]
            for t in range(1, Tc+1):
                m.addConstr(lane_occu[i,move.gdirect,t+1] == lane_occu[i,move.gdirect,t] \
                                    - fI[i,i1,t] + f_star[i,i1,t])
                
        for elem in S:
            i = elem.id
            for t in range(1, Tc+1):
                m.addConstr(f[i,maxcell-1,t] == quicksum(f_star[i,i1,t] for i1 in getExitLinkByApproachLink(i)))      
                
        for i,i1 in movement:
            for t in range(1, Tc+1):
                m.addConstr(fI[i,i1,t] <= lane_occu[i,move.gdirect,t])
                m.addConstr(fI[i,i1,t] <= PQ[i][t] * ratio[i,i1] * volve[i,i1,t]\
                                        * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i])
                m.addConstr(fI[i,i1,t] >= lane_occu[i,move.gdirect,t]-z3[i,i1,t]*M)
                m.addConstr(fI[i,i1,t] >= PQ[i][t] * ratio[i,i1] * volve[i,i1,t]\
                                        * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i]-z4[i,i1,t]*M)

        m.update()
        
        m.Params.TimeLimit = 3600
        
        m.Params.Heuristics = 0.8
        
        m.setObjective(quicksum(lane_occu[i,Movement.idcase[str(i)+','+str(i1)].gdirect,t] for i,i1 in movement
                                                                                for t in range(1,Tc+1)), GRB.MINIMIZE)
                    
        m.optimize()
        m.write('model.lp')
    
        print('\n')
        print('Obj: %g' % m.objVal)
        print('\n')
        
        origin_obj = 0
        for i in S:
            for t in range(1, Tc+1):
                origin_obj += clock*(n[i.id, maxcell-1, t].x - f[i.id, maxcell, t].x)
        print(origin_obj)
        
        delay = m.objVal
        
        df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
        for i,i1 in movement:
            df.loc[str(i)+', '+str(i1)] = np.zeros(Tc)
            for t in range(1, Tc+1):
                df.loc[str(i)+', '+str(i1)][t] = volve[i,i1,t].x
        df.to_csv("output_valve.csv")
        
        density_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
        for i in U:
            for j in range(1,maxcell):
                density_df.loc[str(i)+', '+str(j)] = np.zeros(Tc)
                for t in range(1, Tc+1):
                    density_df.loc[str(i)+', '+str(j)][t] = n[i,j,t].x
        density_df.to_csv("output_density.csv")
        
        flow_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
        for i in U:
            for j in range(1,maxcell):
                flow_df.loc[str(i)+', '+str(j)] = np.zeros(Tc)
                for t in range(1, Tc+1):
                    flow_df.loc[str(i)+', '+str(j)][t] = f[i,j,t].x
        flow_df.to_csv("output_flow.csv")
        
        with open("output_signal.csv", "w") as signalOutput:
            print("from,to,gdirect,theta,phi", file=signalOutput)
            for key in sta:
                elem = Movement.idcase[str(key[0]) + ',' + str(key[1])]
                nd = Link.idcase[key[0]].to_node
                c = cycle_dict[nd]
                print("%s,%i,%f,%f" % (elem.move, elem.gdirect, sta[key].x * c, phi[key].x * c), file=signalOutput)
                
        with open("output_lane_use.csv", 'w') as laneUseOutput:
            print("From,to,lanes", file=laneUseOutput)
            for i,i1 in movement:
                lanenum = 0
                for k in range(1,K[i]+1):
                    lanenum += DeRT[i,i1,k].x
                print(str(i) + ',' + str(i1) + ',' + str(lanenum), file=laneUseOutput)
                
        movement_flow = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
        for i,i1 in movement:
            movement_flow.loc[str(i)+', '+str(i1)] = np.zeros(Tc)
            for t in range(1, Tc+1):
                movement_flow.loc[str(i)+', '+str(i1)][t] = fI[i,i1,t].x
        movement_flow.to_csv('output_movement_flow.csv')
        
        cum_veh_num = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
        for i in U:
            cum_veh_num.loc[str(i)+','+'0'] = np.zeros(Tc)
            cum_veh_num.loc[str(i)+','+'J'] = np.zeros(Tc)
            for t in range(1, Tc+1):
                cum_veh_num.loc[str(i)+','+'0'][t] = N[i,0,t].x
                cum_veh_num.loc[str(i)+','+'J'][t] = N[i,1,t].x
        cum_veh_num.to_csv('Cumulative_Vehicle_Number.csv')
        
        lane_occu_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
        for i,i1 in movement:
            lane_occu_df.loc[str(i)+', '+str(i1)] = np.zeros(Tc)
            for t in range(1, Tc+1):
                lane_occu_df.loc[str(i)+', '+str(i1)][t] = lane_occu[i,Movement.idcase[str(i)+','+str(i1)].gdirect,t].x
        lane_occu_df.to_csv('Lane_Occupancy.csv')
        
        return delay
           
    except AttributeError:
        return 0
        print('Encountered an attribute error')
               
def optimization_CTM(c, isReturnVal=False, isOptimizeCycle=False):
    delay = {}
    density = {}
    results = {}
    start_time = datetime.datetime.now()
    cycle_dict = {3:100, 5:160, 10:100, 11:100}
    try:
        theta_output = {}
        phi_output = {}
        theta_upper_output = {}
        phi_upper_output = {}
        for ndkey in Node.idcase:
            nd = Node.idcase[ndkey]
#            if ndkey != 10:
#                continue

            if nd.type == 0:
                m = Model('si%s'%(nd.id))
                clock = 5
                # gap = clock #clearance time
                gap = 0
                mingreen = 10
                Totaltime = 240
                maxcell = 3
                Tc = int(Totaltime/clock)
                cc = int(c/clock)
                if not isOptimizeCycle:
                    c = cycle_dict[ndkey]
                    cc = int(cycle_dict[ndkey]/clock)
                w = 20   #shock wave speed
                v = 60   #free flow speed
                d = 8    #jam headway in meters
                # K = [0]
                K = {}
                U = []
                S = []
                E = []
                for key in Link.idcase:
                    if Link.idcase[key].from_node == nd.id or Link.idcase[key].to_node == nd.id:
                        # K.append(Link.idcase[key].num_of_lanes)
                        K[Link.idcase[key].id] = Link.idcase[key].num_of_lanes
                        U.append(Link.idcase[key].id)
                
                for key in Node.idcase:
                    if key == nd.id:
                        S.extend(Node.idcase[key].link_approach)
                        E.extend(Node.idcase[key].link_exit)
        
                s = 2105     #saturated flow rate for one lane
                N = {}
                PQ = {}
                
                for i in U:
                    PQ[i] = [(s/3600*clock)*K[i] for t in range(Tc+1)]
                    N[i] = (v/3.6*clock*1/d)*K[i]
            
                M = 1000000 #infinite number
                
                Demand = 1800
                flowinput = [0, (202/3600)*clock, 0, (257/3600)*clock, 0, (207/3600)*clock, 0, (193/3600)*clock]
                initial = 0
                #gdirect = 1 signs left-turn, and 2 right-turn,
                #arm signs intersection arm
                movement_dict = {}
                for key in Movement.idcase:
                    # 排除不属于这个交叉口的路段
                    if Link.idcase[Movement.idcase[key].from_link] not in S:
                        continue

                    movement_dict[(Movement.idcase[key].from_link, Movement.idcase[key].to_link)] \
                        = [Movement.idcase[key].ratio, Movement.idcase[key].gdirect]
                
                movement, ratio, gdirect = multidict(movement_dict)
                conflict_list = initConflict()
                needRemove = []
                for elem in conflict_list:
                    tempFlag = 0
                    for subelem in elem:
                        if Link.idcase[subelem] not in S and Link.idcase[subelem] not in E:
                            tempFlag = 1
                            break
                    if tempFlag:
                        needRemove.append(elem)
                        
                for elem in needRemove:
                    conflict_list.remove(elem)
                    
                conflict_dict = {}
                for elem in conflict_list:
                    conflict_dict[tuple(elem)] = 1
                    
                conflict, value = multidict(conflict_dict)
        
                sta = {}
                phi = {}
                zs = {}
                for i,i1 in movement:
                        sta[i,i1] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='sta%s_%s_'%(i,i1))
                        phi[i,i1] = m.addVar(lb=mingreen/c,ub=1.0,vtype=GRB.CONTINUOUS, name='phi%s_%s'%(i,i1)) 
                        zs[i,i1] = m.addVar(vtype=GRB.BINARY, name='zs%s_%s'%(i,i1))
                omg={}
                for i,i1,i2,i3 in conflict:
                        omg[i,i1,i2,i3]= m.addVar(vtype=GRB.BINARY, name='omg%s_%s_%s_%s'%(i,i1,i2,i3)) 
                
                volve = {}
                fI = {}
                for i,i1 in movement:
                    for t in range(1, Tc+1):
                        fI[i,i1,t] = m.addVar(lb=0.0,vtype=GRB.CONTINUOUS, name='fI%s_%s_%s'%(i,i1,t))
                        volve[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='volve%s_%s_%s'%(i,i1,t))
                zs1={}
                zs2={}
                z3={}
                z4={}
                # zn = {}
            #zs1 and zs2: integer variable for signal settings
            #z3 and z4: integer variable for movement volve
                for i,i1 in movement:
                    for t in range(1, Tc+1):
                        zs1[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs1%s_%s_%s'%(i,i1,t))
                        zs2[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs2%s_%s_%s'%(i,i1,t))
                        z3[i,i1,t]= m.addVar( vtype=GRB.BINARY, name='z3%s_%s_%s'%(i,i1,t))
                        z4[i,i1,t]= m.addVar( vtype=GRB.BINARY, name='z4%s_%s_%s'%(i,i1,t))
                z1={}
                z2={}
                n={}
            # z1 and z2: integer variable for flow constrains
            # n: accumulation
                for i in U:
                    for j in range(1,maxcell):
                        for t in range(1, Tc+2):
                            n[i,j,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='n%s_%s_%s'%(i,j,t))
                            z1[i,j,t]= m.addVar( vtype=GRB.BINARY, name='z1%s_%s_%s'%(i,j,t))
                            z2[i,j,t]= m.addVar( vtype=GRB.BINARY, name='z2%s_%s_%s'%(i,j,t))
                f={}
            #f: flow
                for i in U:
                    for j in range(1,maxcell+1):
                        for t in range(1, Tc+1):
                            f[i,j,t] = m.addVar(lb = 0.0, vtype=GRB.CONTINUOUS, name='f%s_%s_%s'%(i,j,t))
                # initial accumulation is 0
                for i in U:
                    for j in range(1,maxcell):
                        m.addConstr(n[i,j,1] == initial)
            ##link flow input
            
                for elem in S:
                    i = elem.id
                    for t in range(1, Tc+1):
                        m.addConstr(f[i,1,t] <= Link.idcase[i].flowinput[t])
                        m.addConstr(f[i,1,t] >= Link.idcase[i].flowinput[t]-z1[i,1,t]*M)
                            
                        m.addConstr(f[i,1,t] <= PQ[i][t])
                        m.addConstr(f[i,1,t] <= w/v*(N[i]-n[i,1,t]))
                            
                        m.addConstr(f[i,1,t] >= PQ[i][t]-z2[i,1,t]*M)
                        m.addConstr(f[i,1,t] >= w/v*(N[i]-n[i,1,t])-(2-z1[i,1,t]-z2[i,1,t])*M)

            #normal link flow constrains, from cell 2 to the last cell
                for i in U:
                    for j in range(2,maxcell):   
                        for t in range(1, Tc+1):
                            m.addConstr(f[i,j,t] <= n[i,j-1,t])
                            m.addConstr(f[i,j,t] <= PQ[i][t])
                            m.addConstr(f[i,j,t] <= w/v*(N[i]-n[i,j,t]))
                            m.addConstr(f[i,j,t] >= n[i,j-1,t]-z1[i,j,t]*M)
                            m.addConstr(f[i,j,t] >= PQ[i][t]-z2[i,j,t]*M)
                            m.addConstr(f[i,j,t] >= w/v*(N[i]-n[i,j,t])-(2-z1[i,j,t]-z2[i,j,t])*M)
                            
                DeRT={}
                for i,i1 in movement:
                    for k in range(1,K[i]+1):
                        DeRT[i,i1,k]=m.addVar(vtype=GRB.BINARY,name='DeRT%s_%s_%s'%(i,i1,k))

            #1 lane permission
            #2. one lane can share multiple directions
            # constrain 28
                for elem in S:
                    i = elem.id
                    for k in range(1,K[i]+1):
                        m.addConstr(quicksum(DeRT[i,i1,k] for i1 in getExitLinkByApproachLink(i))>=1)
                        
            #share less than two directions
            #3.one movement can distribute on multiple lanes
            # constrain 29
                for i,i1 in movement:
                    m.addConstr(quicksum(DeRT[i,i1,k] for k in range(1,K[i]+1))>=1)
            
            # #4.flow distribution
            # #5. no conflict for adjacent lane settings
                #here 'k+1' constrains that the number of lanes must bigger than one
                # constrain 30
                for elem in S:
                    i = elem.id
                    for i1 in getExitLinkByApproachLink(i):
                        for i2 in getExitLinkByApproachLink(i):
                            for k in range(1,K[i]):
                                if (gdirect[i,i1] > gdirect[i,i2]):
                                    m.addConstr(DeRT[i,i1,k+1]+DeRT[i,i2,k]<=1)
                                    
                for elem in S:
                    i = elem.id
                    for i1 in getExitLinkByApproachLink(i):
                        for i2 in getExitLinkByApproachLink(i):
                            if(ratio[i,i1]>ratio[i,i2]):
                                m.addConstr(quicksum(DeRT[i,i1,k] for k in range(1,K[i]+1))>=quicksum(DeRT[i,i2,k] for k in range(1,K[i]+1)))
            #6.Phase synchronization
                STA={}
                PHI={}
                for elem in S:
                    i = elem.id
                    for k in range(1,K[i]+1):
                        STA[i,k] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='STA%s_%s'%(i,k))
                        PHI[i,k] = m.addVar(lb=mingreen/c,ub=1.0,vtype=GRB.CONTINUOUS, name='PHI%s_%s'%(i,k))
                        
            
            # constrain 37 and 38
                for elem in S:
                    i = elem.id
                    for i1 in getExitLinkByApproachLink(i):
                        for k in range(1,K[i]+1):
                            m.addConstr(M*(1-DeRT[i,i1,k]) >= STA[i,k]-sta[i,i1])
                            m.addConstr(-M*(1-DeRT[i,i1,k]) <= STA[i,k]-sta[i,i1])
                            m.addConstr(M*(1-DeRT[i,i1,k]) >= PHI[i,k]-phi[i,i1])
                            m.addConstr(-M*(1-DeRT[i,i1,k]) <= PHI[i,k]-phi[i,i1])
                        
            #flow into link in E
                for elem in E:
                    i1 = elem.id
                    for t in range(1, Tc+1):
                        m.addConstr(f[i1,1,t] == quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
                        
            #last cell flow on link i in S
                for elem in S:
                    i = elem.id
                    for t in range(1, Tc+1):
                        m.addConstr(f[i,maxcell,t] == quicksum(fI[i,i1,t] for i1 in getExitLinkByApproachLink(i)))
                        
                for i,i1 in movement:
                    m.addConstr(sta[i,i1]+phi[i,i1]<=1)
                    for t in range(1, cc+1):
                        # constrain 14
                        m.addConstr(t >= c/clock * sta[i,i1] - (1 - volve[i,i1,t]) * M)
                        m.addConstr(t <= c/clock * (sta[i,i1] + phi[i,i1]) + (1 - volve[i,i1,t])*M)
                        
                        # constrain 15
                        m.addConstr(t >= c/clock * (sta[i,i1] + phi[i,i1]) - volve[i,i1,t]*M - (1-zs1[i,i1,t]) * M)
                        m.addConstr(t <= c/clock * sta[i,i1] + volve[i,i1,t]* M + zs1[i,i1,t] * M)
                        
            #outflow of link i1 in E
                for elem in E:
                    i1 = elem.id
                    for t in range(1, Tc+1):
                        m.addConstr(f[i1,maxcell,t] == n[i1,maxcell-1,t])
            #CTM update
                for i in U:
                    for j in range(1,maxcell):
                        for t in range(1,Tc+1):
                            m.addConstr(n[i,j,t+1] == n[i,j,t] + f[i,j,t] - f[i,j+1,t])
                        
            #order of movement   
                for i,i1,i2,i3 in conflict:
                    m.addConstr(omg[i,i1,i2,i3]+omg[i2,i3,i,i1]==1)
                    m.addConstr(sta[i,i1]+phi[i,i1]+gap/c <= sta[i2,i3]+omg[i,i1,i2,i3])
        
            #make sure that the value of variables volve is the same in every cycle
                for i,i1 in movement:
                    for t in range(1,Tc+1-cc):
                        m.addConstr(volve[i,i1,t]==volve[i,i1,t+cc])
                    
                # Lock Right turn movement with all green
                for key in movement_dict:
                    if movement_dict[key][1] == 3:
                        m.addConstr(sta[key[0], key[1]] == 0)
                        m.addConstr(phi[key[0], key[1]] == 1)
                    
                # Coordination
                for elem in E:
                    i1 = elem.id
                    for t in range(1, Tc+1):
                        m.addConstr(w/v*(N[i1] - n[i1,1,t]) >= quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
                        
                lane_occu = {}
                for elem in S:
                    i = elem.id
                    for k in range(1, 4):
                        for t in range(Tc+2):
                            lane_occu[i,k,t] = m.addVar(lb = 0, vtype=GRB.CONTINUOUS, name='lb%s_%s_%i'%(i,k,t))
                            
                f_star = {}
                for elem in S:
                    i = elem.id
                    exits = getExitLinkByApproachLink(i)
                    for i1 in exits:
                        for t in range(Tc+2):
                            f_star[i,i1,t] = m.addVar(lb = 0, vtype=GRB.CONTINUOUS, name='f_star%s%s%i' % (i,i1,t))
                            
                zs3={}
                zs4={}

                for i,i1 in movement:
                    for t in range(1, Tc+1):
                        zs3[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs3%s_%s_%s'%(i,i1,t))
                        zs4[i,i1,t] = m.addVar(vtype=GRB.BINARY, name='zs4%s_%s_%s'%(i,i1,t))
                        
                for elem in S:
                    i = elem.id
                    exits = getExitLinkByApproachLink(i)
                    for i1 in exits:
                        move = Movement.idcase[str(i)+','+str(i1)]
                        for t in range(1, Tc+1):
                            if maxcell <= 2:    
                                m.addConstr(f_star[i,i1,t] <= Link.idcase[i].flowinput[t] * ratio[i,i1])
                                m.addConstr(f_star[i,i1,t] >= Link.idcase[i].flowinput[t] * ratio[i,i1]-zs3[i,i1,t]*M)
                            else:
                                m.addConstr(f_star[i,i1,t] <= n[i,maxcell-2,t] * ratio[i,i1])
                                m.addConstr(f_star[i,i1,t] >= n[i,maxcell-2,t] * ratio[i,i1]-zs3[i,i1,t]*M)
                                
                            m.addConstr(f_star[i,i1,t] <= PQ[i][t] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i])
                            m.addConstr(f_star[i,i1,t] <= w/v*(N[i] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i] - lane_occu[i,move.gdirect,t]))
                                
                            m.addConstr(f_star[i,i1,t] >= PQ[i][t] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i] - zs4[i,i1,t]*M)
                            m.addConstr(f_star[i,i1,t] >= w/v*(N[i] * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i] - lane_occu[i,move.gdirect,t]) - (2-zs3[i,i1,t]-zs4[i,i1,t])*M)
                            
                for i,i1 in movement:
                    move = Movement.idcase[str(i)+','+str(i1)]
                    for t in range(1, Tc+1):
                        m.addConstr(lane_occu[i,move.gdirect,t+1] == lane_occu[i,move.gdirect,t] \
                                    - fI[i,i1,t] + f_star[i,i1,t])
                        
                for i,i1 in movement:
                    move = Movement.idcase[str(i)+','+str(i1)]
                    for t in range(1, Tc+1):
                        m.addConstr(fI[i,i1,t] <= lane_occu[i,move.gdirect,t])
                        m.addConstr(fI[i,i1,t] <= PQ[i][t] * ratio[i,i1] * volve[i,i1,t]\
                                    * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i])
                        m.addConstr(fI[i,i1,t] <= w/v*(N[i1]-n[i1,1,t]))
                        m.addConstr(fI[i,i1,t] >= lane_occu[i,move.gdirect,t] - z3[i,i1,t]*M)
                        m.addConstr(fI[i,i1,t] >= PQ[i][t] * ratio[i,i1] * volve[i,i1,t]\
                                    * quicksum(DeRT[i,i1,k] for k in range(1, K[i]+1)) / K[i] -z4[i,i1,t]*M)
                        m.addConstr(fI[i,i1,t] >= w/v * (N[i1]-n[i1,1,t]) - (2-z3[i,i1,t]-z4[i,i1,t])*M)
                        
                m.update()
                
                m.Params.TimeLimit = 2400
                    
                m.Params.Heuristics = 0.8
                
                m.setObjective(quicksum(lane_occu[i,Movement.idcase[str(i)+','+str(i1)].gdirect,t] for i,i1 in movement
                                                                                for t in range(1,Tc+1)), GRB.MINIMIZE)

                m.write('model.lp')            
                m.optimize()
                
                origin_obj = 0
                tempdens = 0
                for i in S:
                    for t in range(1, Tc+1):
                        origin_obj += clock*(n[i.id, maxcell-1, t].x - f[i.id, maxcell, t].x)
                        tempdens += clock*(n[i.id, maxcell-1, t].x)
                print(origin_obj)
                
                delay[ndkey] = origin_obj
                density[ndkey] = m.objVal
                
                m.write("model.lp")
                
                df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
                for i,i1 in movement:
                    df.loc[str(i)+', '+str(i1)] = np.zeros(Tc)
                    for t in range(1, Tc+1):
                        df.loc[str(i)+', '+str(i1)][t] = volve[i,i1,t].x
                df.to_csv("output_valve.csv")
                
                density_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
                for i in U:
                    for j in range(1,maxcell):
                        density_df.loc[str(i)+', '+str(j)] = np.zeros(Tc)
                        for t in range(1, Tc+1):
                            density_df.loc[str(i)+', '+str(j)][t] = n[i,j,t].x
                density_df.to_csv("output_density.csv")
                
                flow_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
                for i in U:
                    for j in range(1,maxcell):
                        flow_df.loc[str(i)+', '+str(j)] = np.zeros(Tc)
                        for t in range(1, Tc+1):
                            flow_df.loc[str(i)+', '+str(j)][t] = f[i,j,t].x
                flow_df.to_csv("output_flow.csv")
                
                movement_flow = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
                for i,i1 in movement:
                    movement_flow.loc[str(i)+', '+str(i1)] = np.zeros(Tc)
                    for t in range(1, Tc+1):
                        movement_flow.loc[str(i)+', '+str(i1)][t] = fI[i,i1,t].x
                movement_flow.to_csv('output_movement_flow.csv')
                
                lane_occu_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
                for i,i1 in movement:
                    lane_occu_df.loc[str(i)+', '+str(i1)] = np.zeros(Tc)
                    for t in range(1, Tc+1):
                        lane_occu_df.loc[str(i)+', '+str(i1)][t] = lane_occu[i,Movement.idcase[str(i)+','+str(i1)].gdirect,t].x
                lane_occu_df.to_csv('Lane_Occupancy.csv')
                
                for key in sta:
                    theta_output[key] = sta[key].x
                    
                for key in phi:
                    phi_output[key] = phi[key].x
                    
                for key in STA:
                    theta_upper_output[key] = STA[key].x
                    
                for key in PHI:
                    phi_upper_output[key] = PHI[key].x
                    
                with open("output_signal.csv", "w") as signalOutput:
                    print("from,to,gdirect,theta,phi", file=signalOutput)
                    for key in sta:
                        elem = Movement.idcase[str(key[0]) + ',' + str(key[1])]
                        print("%s,%i,%f,%f" % (elem.move, elem.gdirect, sta[key].x * c, phi[key].x * c), file=signalOutput)
        
        end_time = datetime.datetime.now()
        print(end_time - start_time)
        if isReturnVal:
            return (delay, density) 
        else:
            return (theta_output, phi_output, theta_upper_output, phi_upper_output)
        
    except AttributeError:
        print('Encountered an attribute error')

def getOptimalCycle():
    delay = {}
    density = {}
    for ndkey in Node.idcase:
        nd = Node.idcase[ndkey]

        if nd.type == 0:
            delay[ndkey] = {}
            density[ndkey] = {}
    for i in range(100, 200, 10):
        result = optimization_CTM(i, isReturnVal=True, isOptimizeCycle=True)
        temp_delay = result[0]
        temp_density = result[1]
        for key in temp_delay:
            delay[key][i] = temp_delay[key]
            density[key][i] = temp_density[key]
    
    for key in delay:
        print(key, min(delay[key], key=delay[key].get))
        
    for key in density:
        print(key, min(density[key], key=density[key].get))
        
def getOptimalCycle_Network_GA():
    random.seed(123)
    population = []
    init_chromo = [100 for i in range(4)]
    next_gene = []
    solutions = []
    best_chromo = []
    # chromo_mapping = {5:0 * totalTime, 6:1 * totalTime, 4:2 * totalTime, 8:3 * totalTime}
    results = []
    for k in range(1000):
        if k:
            population = next_gene
            next_gene = []
        else:
            for i in range(10):
                if i:
                    newpop = pop(population[0].chromo.copy())
                    newpop.mutation()
                    population.append(newpop)
                else:
                    population.append(pop(init_chromo))
            
        for elem in population:
            elem.get_fitness()
                
        nfl = getNewFitnessList(population)
        best = min(nfl, key=nfl.get)
        best_chromo.append(best.chromo.copy())
        # print("Iterations:", k, '\n', "Best Fitness:", best.fitness, '\n', "Best Chromo:", best.chromo)
        print("Iterations:", k, '\n', "Best Fitness:", best)
        # ziliaskWithSCTMC_New(best.chromo, isReturn=False)
        print("Standard Deviation:", getSD())
                
        for j in range(9):
            next_gene.append(selection(population))
                
            # for m in range(5):
            #     next_gene.append(random.choice(next_gene[0:5]).crossover(random.choice(next_gene[5:10])))
            
        temp_cross_list = []
        for elem in next_gene:
            probOfCross = 0.88
            if random.random() <= probOfCross:
                temp_cross_list.append(elem)
                
            if len(temp_cross_list) > 1:
                if temp_cross_list[0].chromo == temp_cross_list[1].chromo:
                    temp_cross_list.pop()
                    temp_pop = pop(temp_cross_list[0].chromo.copy())
                    temp_pop.mutation()
                    while temp_pop.chromo == temp_cross_list[0].chromo:
                        temp_pop.mutation()
                            
                    temp_cross_list.append(temp_pop)
                    
                temp_cross_list[0].newCrossover(temp_cross_list[1])
                temp_cross_list = []
                    
        temp_idx = 100
        for elem in next_gene:
            solutions.append(elem.chromo.copy())
                
        for n in range(6):
            temp_mut = random.choice(next_gene)
            if temp_idx == next_gene.index(temp_mut):
                continue
            temp_idx = next_gene.index(temp_mut)
            temp_mut.mutation()
            while temp_mut.chromo in solutions:
                temp_mut.mutation()
            
        next_gene.append(pop(best_chromo[-1]))
        results.append(best.fitness)
                
    # print("Objective:", ziliaskWithSCTMC_New(best_chromo[-1]))
    print("Standard Deviation:", getSD())
    print(best_chromo[-1])
    # plotProportion(best.chromo)
    # plotMFD()
            
if __name__ == '__main__':
    np.random.seed(123)
    random.seed(123)
    readNetwork()
    initMovement()
    calibrate_all()
    # Movement.getAllMovement()
    # calibrateFlowInput()
    # calibrateRatio()
    # initConflict()
#    getOptimalCycle()
    start = datetime.datetime.now()
    lp_optimize(100, *optimization_CTM(100))
    end = datetime.datetime.now()
    print(end - start)
    # getOptimalCycle_Network_GA()
#    optimization_CTM(100)
    # lp_optimize(100)
    # output = optimization_distributed(130)
    # lp_optimize(100, 1, input_chromo=[100, 100, 100, 100])
    # print("theta:", output[0], '\n')
    # print("phi:", output[1], '\n')
    # print("THETA:", output[2], '\n')
    # print("PHI:", output[3], '\n')
    # lp_optimize(200, 1)