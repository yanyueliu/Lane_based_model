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
    def __init__(self, lid, from_nd, to_nd, vf, num_of_lanes, w=12, kj=120, qmax=1800, flow_input=200):
        self.id = lid
        self.from_node = from_nd
        self.to_node = to_nd
        self.num_of_lanes = num_of_lanes
        self.vf = vf
        self.w = w
        self.kj = kj
        self.qmax = qmax
        self.flowinput = flow_input
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
        
def readNetwork():
    link_file = pd.read_csv("link_xueyuan_9nodes.csv")
    node_file = pd.read_csv("node_xueyuan_9nodes.csv")
    
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
                    if to_link == other_to_link:
                        continue
                    
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
        if Movement.idcase[key].from_link == n:
            tempList.append(Movement.idcase[key].to_link)
            
    return tempList

def getApproachLinkByExitLink(n):
    tempList = []
    for key in Movement.idcase:
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
            Link.idcase[key].flowinput = sum(flow_input[key])

def lp_optimize(c, bina, init_theta={}, init_phi={}, init_theta_upper={}, init_phi_upper={}):
    try:
        m = Model('si%i'%(c))
        clock = 5
        gap = clock #clearance time
        mingreen = 15
        Totaltime = 2.5*c
        maxcell = 2
        Tc = int(Totaltime/clock)
        cc = int(c/clock)
        w = 12   #shock wave speed
        v = 60   #free flow speed
        vf = 60
        d = 5    #jam headway in meters
        K = [0]
        U = []
        S = []
        E = []
        for key in Link.idcase:
            K.append(Link.idcase[key].num_of_lanes)
            U.append(Link.idcase[key].id)
        
        for key in Node.idcase:
            S.extend(Node.idcase[key].link_approach)
            E.extend(Node.idcase[key].link_exit)

        s = 2200     #saturated flow rate for one lane
        PQ, Kjam = [], []
        for i in range(len(K)):
            PQ.append((s/3600*clock)*K[i])
            Kjam.append((v/3.6*clock*1/d)*K[i])
    
        M = 100000 #infinite number
        
        Demand = 1800
        # flowinput = [0, (202/3600)*clock, 0, (257/3600)*clock, 0, (207/3600)*clock, 0, (193/3600)*clock]
        initial = 0
        #gdirect = 1 signs left-turn, and 2 right-turn,
        #arm signs intersection arm
        movement_dict = {}
        for key in Movement.idcase:
            # if Movement.idcase[key].gdirect == 1:
            #     Movement.idcase[key].ratio = 0.2
            # elif Movement.idcase[key].gdirect == 2:
            #     Movement.idcase[key].ratio = 0.7
            # else:
            #     Movement.idcase[key].ratio = 0.1
                
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
        
        if len(init_theta) == 0 and len(init_phi) == 0:
            for i,i1 in movement:
                sta[i,i1] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='sta%s_%s_'%(i,i1))
                phi[i,i1] = m.addVar(lb=mingreen/c,ub=1.0,vtype=GRB.CONTINUOUS, name='phi%s_%s'%(i,i1)) 
                zs[i,i1] = m.addVar(vtype=GRB.BINARY, name='zs%s_%s'%(i,i1))
            
            for elem in S:
                i = elem.id
                
                for k in range(1,K[i]+1):
                    STA[i,k] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='STA%s_%s'%(i,k))
                    PHI[i,k] = m.addVar(lb=mingreen/c,ub=1.0,vtype=GRB.CONTINUOUS, name='PHI%s_%s'%(i,k))
        else:
            for i,i1 in movement:
                sta[i,i1] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='sta%s_%s_'%(i,i1))
                phi[i,i1] = m.addVar(lb=mingreen/c,ub=1.0,vtype=GRB.CONTINUOUS, name='phi%s_%s'%(i,i1)) 
                zs[i,i1] = m.addVar(vtype=GRB.BINARY, name='zs%s_%s'%(i,i1))
                sta[i,i1].start = init_theta[i,i1]
                phi[i,i1].start = init_phi[i,i1]
                # m.addConstr(sta[i,i1] == init_theta[i,i1])
                m.addConstr(phi[i,i1] == init_phi[i,i1])
            for elem in S:
                i = elem.id
                
                for k in range(1,K[i]+1):
                    STA[i,k] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='STA%s_%s'%(i,k))
                    PHI[i,k] = m.addVar(lb=mingreen/c,ub=1.0,vtype=GRB.CONTINUOUS, name='PHI%s_%s'%(i,k))
                    STA[i,k].start = init_theta_upper[i,k]
                    PHI[i,k].start = init_phi_upper[i,k]
                    # m.addConstr(STA[i,k] == init_theta_upper[i,k])
                    m.addConstr(PHI[i,k] == init_phi_upper[i,k])
                
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
    # Integrate new variables
    #     m.update()
    
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
                        
        for elem in E:
            i = elem.id
                    
            for t in range(1, Tc+1):
                linkLength = getEuclideanDis(Node.idcase[Link.idcase[i].from_node].x, 
                                             Node.idcase[Link.idcase[i].to_node].x, 
                                             Node.idcase[Link.idcase[i].from_node].y, 
                                             Node.idcase[Link.idcase[i].to_node].y) / 1000
                        
                bwtt = t + 1 - int((linkLength / w) / clock * 3600)
                fftt = t + 1 - int((linkLength / vf) / clock * 3600)
                        
                if bwtt <= 0:
                    m.addConstr(Receiving[i,t] <= 0 + Kjam[i] * linkLength - N[i,0,t])
                    m.addConstr(Receiving[i,t] >= 0 + Kjam[i] * linkLength - N[i,0,t] - M * zx1[i,t])
                else:
                    m.addConstr(Receiving[i,t] <= N[i,1,bwtt] + Kjam[i] * linkLength - N[i,0,t])
                    m.addConstr(Receiving[i,t] >= N[i,1,bwtt] + Kjam[i] * linkLength - N[i,0,t] - M * zx1[i,t])
                            
                m.addConstr(Receiving[i,t] <= PQ[i])
                m.addConstr(Receiving[i,t] >= PQ[i] - M * (1 - zx1[i,t]))
                        
                m.addConstr(G[i,t] <= f[i,2,t] - N[i,0,t-1])
                m.addConstr(G[i,t] >= f[i,2,t] - N[i,0,t-1] - M * zx2[i,t])
                m.addConstr(G[i,t] <= Receiving[i,t])
                m.addConstr(G[i,t] >= Receiving[i,t] - M * (1 - zx2[i,t]))
                        
                m.addConstr(N[i,0,t] == N[i,0,t-1] + G[i,t])
                        
                if fftt <= 0:
                    m.addConstr(Sending[i,t] <= 0 - N[i,1,t])
                    m.addConstr(Sending[i,t] >= 0 - N[i,1,t] - M * zx3[i,t])
                else:
                    m.addConstr(Sending[i,t] <= N[i,0,fftt] - N[i,1,t])
                    m.addConstr(Sending[i,t] >= N[i,0,fftt] - N[i,1,t] - M *zx3[i,t])
                            
                m.addConstr(Sending[i,t] <= PQ[i])
                m.addConstr(Sending[i,t] >= PQ[i] - M * (1 - zx3[i,t]))
                m.addConstr(N[i,1,t] == N[i,1,t-1] + Sending[i,t])
                            
            for elem in S:
                i = elem.id
                    
                for t in range(1, Tc+1):
                    linkLength = getEuclideanDis(Node.idcase[Link.idcase[i].from_node].x, 
                                                 Node.idcase[Link.idcase[i].to_node].x, 
                                                 Node.idcase[Link.idcase[i].from_node].y, 
                                                 Node.idcase[Link.idcase[i].to_node].y) / 1000
                        
                    bwtt = t + 1 - int((linkLength / w) / clock * 3600)
                    fftt = t + 1 - int((linkLength / vf) / clock * 3600)
                        
                    if bwtt <= 0:
                        m.addConstr(Receiving[i,t] <= 0 + Kjam[i] * linkLength - N[i,0,t])
                        m.addConstr(Receiving[i,t] >= 0 + Kjam[i] * linkLength - N[i,0,t] - M * zx1[i,t])
                    else:
                        m.addConstr(Receiving[i,t] <= N[i,1,bwtt] + Kjam[i] * linkLength - N[i,0,t])
                        m.addConstr(Receiving[i,t] >= N[i,1,bwtt] + Kjam[i] * linkLength - N[i,0,t] - M * zx1[i,t])
                            
                    m.addConstr(Receiving[i,t] <= PQ[i])
                    m.addConstr(Receiving[i,t] >= PQ[i] - M * (1 - zx1[i,t]))
                        
                    m.addConstr(G[i,t] <= elem.flowinput * 12 / 3600 * clock - N[i,0,t-1])
                    m.addConstr(G[i,t] >= elem.flowinput * 12 / 3600 * clock - N[i,0,t-1] - M * zx2[i,t])
                    m.addConstr(G[i,t] <= Receiving[i,t])
                    m.addConstr(G[i,t] >= Receiving[i,t] - M * (1 - zx2[i,t]))
                        
                    m.addConstr(N[i,0,t] == N[i,0,t-1] + G[i,t])
                        
                    if fftt <= 0:
                        m.addConstr(Sending[i,t] <= 0 - N[i,1,t])
                        m.addConstr(Sending[i,t] >= 0 - N[i,1,t] - M * zx3[i,t])
                    else:
                        m.addConstr(Sending[i,t] <= N[i,0,fftt] - N[i,1,t])
                        m.addConstr(Sending[i,t] >= N[i,0,fftt] - N[i,1,t] - M *zx3[i,t])
                            
                    m.addConstr(Sending[i,t] <= PQ[i])
                    m.addConstr(Sending[i,t] >= PQ[i] - M * zx4[i,t])
                        
                    m.addConstr(Sending[i,t] <= w/v * (Kjam[i] - n[i,1,t]))
                    m.addConstr(Sending[i,t] >= w/v * (Kjam[i] - n[i,1,t]) - M * (2 - zx3[i,t] - zx4[i,t]))
                    m.addConstr(N[i,1,t] == N[i,1,t-1] + Sending[i,t])
                    m.addConstr(f[i,1,t] == Sending[i,t])
     
    #CTM update
        for i in U:
            for j in range(1,maxcell):
                for t in range(1,Tc+1):
                    m.addConstr(n[i,j,t+1] == n[i,j,t] + f[i,j,t] - f[i,j+1,t])
    # ##link flow input
    #     for elem in S:
    #         i = elem.id
    #         for t in range(1, Tc+1):
    #             m.addConstr(f[i,1,t] <= n[i,1,t])
    #             m.addConstr(f[i,1,t] >= n[i,1,t]-z1[i,1,t]*M)
                    
    #             m.addConstr(f[i,1,t] <= PQ[i])
    #             m.addConstr(f[i,1,t] <= w/v*(Kjam[i]-n[i,1,t]))
                
    #             m.addConstr(f[i,1,t] >= PQ[i]-z2[i,1,t]*M)
    #             m.addConstr(f[i,1,t] >= w/v*(Kjam[i]-n[i,1,t])-(2-z1[i,1,t]-z2[i,1,t])*M)

        #flow into link in E
        for elem in E:
            i1 = elem.id
            # if Node.idcase[elem.to_node].type == 0:
            #     continue
            for t in range(1, Tc+1):
                m.addConstr(f[i1,1,t] == quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
                
        # Coordination
        for elem in E:
            i1 = elem.id
            for t in range(1, Tc+1):
                m.addConstr(w/v*(Kjam[i1] - n[i1,1,t]) >= quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
                
        # Coordination
        # for elem in E:
        #     i1 = elem.id
        #     for t in range(1, Tc+1):
        #         linkLength = getEuclideanDis(Node.idcase[Link.idcase[i].from_node].x, 
        #                                      Node.idcase[Link.idcase[i].to_node].x, 
        #                                      Node.idcase[Link.idcase[i].from_node].y, 
        #                                      Node.idcase[Link.idcase[i].to_node].y)
                
        #         bwtt = t + 1 - int((linkLength / w) / clock)
        #         if bwtt <= 0:
        #             m.addConstr(Kjam[i] * linkLength - N[i,0,t] >= quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
        #         else:
        #             m.addConstr(N[i,1,bwtt] + Kjam[i] * linkLength - N[i,0,t] >= quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
                
                
        #last cell flow on link i in S
        for elem in S:
            i = elem.id
            for t in range(1, Tc+1):
                m.addConstr(f[i,maxcell,t] == quicksum(fI[i,i1,t] for i1 in getExitLinkByApproachLink(i)))
        
    #movement flow constrains
        for i,i1 in movement:
            for t in range(1, Tc+1):
                m.addConstr(fI[i,i1,t]<=n[i,maxcell-1,t]*ratio[i,i1])
                m.addConstr(fI[i,i1,t]<=PQ[i]*ratio[i,i1]*volve[i,i1,t])
                m.addConstr(fI[i,i1,t]<=w/v*(Kjam[i]-n[i1,1,t]))
                m.addConstr(fI[i,i1,t]>=n[i,maxcell-1,t]*ratio[i,i1]-z3[i,i1,t]*M)
                m.addConstr(fI[i,i1,t]>=PQ[i]*ratio[i,i1]*volve[i,i1,t]-z4[i,i1,t]*M)
                m.addConstr(fI[i,i1,t]>=w/v*(Kjam[i]-n[i1,1,t])-(2-z3[i,i1,t]-z4[i,i1,t])*M)
                
    #outflow of link i1 in E
        # for elem in E:
        #     i1 = elem.id
        #     if Node.idcase[elem.to_node].type == 0:
        #         continue
        #     for t in range(1, Tc+1):
        #         m.addConstr(f[i1,maxcell,t] == n[i1,maxcell-1,t])
    
                
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
                m.addConstr(quicksum(DeRT[i,i1,k] for i1 in getExitLinkByApproachLink(i))<=2)
                
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
    #6.Phase synchronization
        # STA={}
        # PHI={}
        # for elem in S:
        #     i = elem.id
        #     for k in range(1,K[i]+1):
        #         STA[i,k] = m.addVar(lb=0.0,ub=0.9,vtype=GRB.CONTINUOUS, name='STA%s_%s'%(i,k))
        #         PHI[i,k] = m.addVar(lb=mingreen/c,ub=1.0,vtype=GRB.CONTINUOUS, name='PHI%s_%s'%(i,k))
                
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
            m.addConstr(sta[i,i1]+phi[i,i1]<=1+zs[i,i1]*M)
            m.addConstr(1.01<=sta[i,i1]+phi[i,i1]+(1-zs[i,i1])*M)
            for t in range(1, cc+1):
                # constrain 14
                m.addConstr(t >= c/clock * sta[i,i1] - zs[i,i1] * M - (1 - volve[i,i1,t]) * M)
                m.addConstr(t <= c/clock * (sta[i,i1] + phi[i,i1]) + (1 - volve[i,i1,t])*M + zs[i,i1]*M)
                
                # constrain 15
                m.addConstr(t >= c/clock * (sta[i,i1] + phi[i,i1]) - volve[i,i1,t]*M - zs[i,i1]*M - (1-zs1[i,i1,t]) * M)
                m.addConstr(t <= c/clock * sta[i,i1] + volve[i,i1,t]* M + zs[i,i1]*M + zs1[i,i1,t] * M)
                
                # constrain 17
                m.addConstr(t >= c/clock * (sta[i,i1] + phi[i,i1] - 1) - (1 - zs[i,i1]) * M - volve[i,i1,t] * M)
                m.addConstr(t <= c/clock * sta[i,i1] + volve[i,i1,t] * M + (1 - zs[i,i1]) * M)
                
                # constrain 18
                m.addConstr(t >= c/clock * sta[i,i1] - (1 - zs[i,i1])*M - (1 - volve[i,i1,t])*M - (1 - zs2[i,i1,t])*M)
                m.addConstr(t <= c/clock * (sta[i,i1] + phi[i,i1] - 1) + (1 - volve[i,i1,t])*M + (1 - zs[i,i1])*M + zs2[i,i1,t]*M)
                
    #order of movement   
        for i,i1,i2,i3 in conflict:
            m.addConstr(omg[i,i1,i2,i3]+omg[i2,i3,i,i1]==1)
            m.addConstr(sta[i,i1]+phi[i,i1]+gap/c <= sta[i2,i3]+omg[i,i1,i2,i3])
            # for k1 in range(1, K[i]+1):
            #     for k2 in range(1, K[i1]+1):
            #         m.addConstr(sta[i,i1]+phi[i,i1]+gap/c <= sta[i2,i3]+omg[i,i1,i2,i3] + M*(2-DeRT[i,i1,k1]-DeRT[i2,i3,k2]))

    #make sure that the value of variables volve is the same in every cycle
        for i,i1 in movement:
            for t in range(1,Tc+1-cc):
                m.addConstr(volve[i,i1,t]==volve[i,i1,t+cc])
                
    # total output
        OP={}
        for elem in E:
            i = elem.id
            if Node.idcase[elem.to_node].type == 0:
                continue
            for t in range(1,Tc+1):
                OP=m.addVar(vtype=GRB.CONTINUOUS,name='OP%s_%s'%(i,t))
                m.addConstr(OP==quicksum(f[i.id,maxcell,t] for i in E
                                                            for t in range(1,Tc+1)))
                
        D={}
        for i in U:
            for t in range(1,Tc+1):
                D[i,t]=m.addVar(vtype=GRB.CONTINUOUS,name='D%s_%s'%(i,t))
    
        for i in U:
            for t in range(1,Tc+1):
                m.addConstr(D[i,t]==clock*quicksum(n[i,j,t]-f[i,j+1,t] for j in range(1,maxcell)))
        DT={}
        for t in range(1,Tc+1):
            DT[t]=m.addVar(vtype = GRB.CONTINUOUS,name='DT%s'%(t))
        for t in range(1,Tc+1):
            m.addConstr(DT[t] == quicksum(D[i,t] for i in U))
            
        
            
        if len(init_theta) == 0 and len(init_phi) == 0:
            for key in Movement.idcase:
                if Movement.idcase[key].gdirect != 2:
                    continue
                from_link = Link.idcase[Movement.idcase[key].from_link]
                fl_vector = [Node.idcase[from_link.to_node].x - Node.idcase[from_link.from_node].x, 
                          Node.idcase[from_link.to_node].y - Node.idcase[from_link.from_node].y]
                
                if fl_vector[1] != 0:
                    if abs(fl_vector[0] / fl_vector[1]) <= 0.333:
                        m.addConstr(sta[Movement.idcase[key].from_link, Movement.idcase[key].to_link] == 0)
                    
        m.update()
    #Set objective
        # m.setObjective(clock*quicksum(n[i.id,maxcell-1,t]-f[i.id,maxcell,t] for i in S
        #                                                                 for t in range(1,Tc+1)), GRB.MINIMIZE)
        
        # m.setObjective(clock*quicksum(-f[i.id,j,t] for i in S for j in range(1, maxcell)
        #                              for t in range(1,Tc+1)), GRB.MINIMIZE)
        
        m.Params.TimeLimit = 600
        
        # m.Params.tuneResults = 1
        # m.Params.tuneTimeLimit = 7200
        # m.tune()
        # if m.tuneResultCount > 0:
        #     # Load the best tuned parameters into the model
        #     m.getTuneResult(0)
        #     # Write tuned parameters to a file
        #     m.write('tune_integer.prm')
        
        m.Params.Heuristics = 0.8
        # m.Params.Method = 1
        
        m.setObjective(clock*quicksum(n[i.id,j,t] for i in S for j in range(1,maxcell)
                                                                        for t in range(1,Tc+1)), GRB.MINIMIZE)
                    
        m.optimize()
        m.write('ergodic_cycle_from_45_to_180\model.mps')
        m.write('ergodic_cycle_from_45_to_180\model.rew')
        m.write('ergodic_cycle_from_45_to_180\model.lp')
        m.write('ergodic_cycle_from_45_to_180\model.rlp')
    
        print('\n')
        print('Obj: %g' % m.objVal)
        print('cycle length:', c)
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
           
    except AttributeError:
        print('Encountered an attribute error')
        
def optimization_distributed(c):
    try:
        theta_output = {}
        phi_output = {}
        theta_upper_output = {}
        phi_upper_output = {}
        for ndkey in Node.idcase:
            nd = Node.idcase[ndkey]
            if nd.id != 17:
                continue
            if nd.type == 0:
                m = Model('si%i'%(nd.id))
                clock = 5
                gap = clock #clearance time
                mingreen = 15
                Totaltime = 2.5*c
                maxcell = 2
                Tc = int(Totaltime/clock)
                cc = int(c/clock)
                w = 12   #shock wave speed
                v = 60   #free flow speed
                vf = v
                d = 5    #jam headway in meters
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
        
                s = 2200     #saturated flow rate for one lane
                Kjam = {}
                PQ = {}
                
                for i in U:
                    PQ[i] = (s/3600*clock)*K[i]
                    Kjam[i] = (v/3.6*clock*1/d)*K[i]
            
                M = 1000000 #infinite number
                
                Demand = 1800
                flowinput = [0, (202/3600)*clock, 0, (257/3600)*clock, 0, (207/3600)*clock, 0, (193/3600)*clock]
                initial = 0
                #gdirect = 1 signs left-turn, and 2 right-turn,
                #arm signs intersection arm
                movement_dict = {}
                for key in Movement.idcase:
                    # if Movement.idcase[key].gdirect == 1:
                    #     Movement.idcase[key].ratio = 0.2
                    # elif Movement.idcase[key].gdirect == 2:
                    #     Movement.idcase[key].ratio = 0.7
                    # else:
                    #     Movement.idcase[key].ratio = 0.1
                    
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
                space = {}
            # z1 and z2: integer variable for flow constrains
            # n: accumulation
                for i in U:
                    for j in range(1,maxcell):
                        for t in range(1, Tc+2):
                            n[i,j,t] = m.addVar(lb = 0.0, ub = Kjam[i], vtype=GRB.CONTINUOUS, name='n%s_%s_%s'%(i,j,t))
                            z1[i,j,t]= m.addVar( vtype=GRB.BINARY, name='z1%s_%s_%s'%(i,j,t))
                            z2[i,j,t]= m.addVar( vtype=GRB.BINARY, name='z2%s_%s_%s'%(i,j,t))
                            space[i,j,t] = m.addVar(vtype=GRB.BINARY, name='space%s_%s_%s'%(i,j,t))
                f={}
            #f: flow
                for i in U:
                    for j in range(1,maxcell+1):
                        for t in range(1, Tc+1):
                            f[i,j,t] = m.addVar(lb = 0.0, ub = PQ[i], vtype=GRB.CONTINUOUS, name='f%s_%s_%s'%(i,j,t))
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
                    for t in range(Tc+1):
                        m.addConstr(zx3[i,t] + zx4[i,t] + zx5[i,t] == 2)
                        
                for elem in E:
                    i = elem.id
                    
                    for t in range(1, Tc+1):
                        linkLength = getEuclideanDis(Node.idcase[Link.idcase[i].from_node].x, 
                                                      Node.idcase[Link.idcase[i].to_node].x, 
                                                      Node.idcase[Link.idcase[i].from_node].y, 
                                                      Node.idcase[Link.idcase[i].to_node].y) / 1000
                        
                        bwtt = t - int((linkLength / w) / clock * 3600)
                        fftt = t - int((linkLength / vf) / clock * 3600)
                        
                        if bwtt <= 0:
                            m.addConstr(Receiving[i,t-1] <= 0 + 120*K[i] * linkLength - N[i,0,t-1])
                            m.addConstr(Receiving[i,t-1] >= 0 + 120*K[i] * linkLength - N[i,0,t-1] - M * zx1[i,t])
                        else:
                            m.addConstr(Receiving[i,t-1] <= N[i,1,bwtt] + 120*K[i] * linkLength - N[i,0,t-1])
                            m.addConstr(Receiving[i,t-1] >= N[i,1,bwtt] + 120*K[i] * linkLength - N[i,0,t-1] - M * zx1[i,t])
                            
                        m.addConstr(Receiving[i,t-1] <= PQ[i])
                        m.addConstr(Receiving[i,t-1] >= PQ[i] - M * (1 - zx1[i,t]))
                        
                        m.addConstr(G[i,t-1] <= quicksum(f[i,2,t1] for t1 in range(1, t)) - N[i,0,t-1])
                        m.addConstr(G[i,t-1] >= quicksum(f[i,2,t1] for t1 in range(1, t)) - N[i,0,t-1] - M * zx2[i,t])
                        m.addConstr(G[i,t-1] <= Receiving[i,t-1])
                        m.addConstr(G[i,t-1] >= Receiving[i,t-1] - M * (1 - zx2[i,t]))
                        
                        m.addConstr(N[i,0,t] == N[i,0,t-1] + G[i,t-1])
                        
                        if fftt <= 0:
                            m.addConstr(Sending[i,t-1] <= 0 - N[i,1,t])
                            m.addConstr(Sending[i,t-1] >= 0 - N[i,1,t] - M * zx3[i,t])
                        else:
                            m.addConstr(Sending[i,t-1] <= N[i,0,fftt] - N[i,1,t-1])
                            m.addConstr(Sending[i,t-1] >= N[i,0,fftt] - N[i,1,t-1] - M *zx3[i,t])
                            
                        m.addConstr(Sending[i,t-1] <= PQ[i])
                        m.addConstr(Sending[i,t-1] >= PQ[i] - M * (1 - zx3[i,t]))
                        m.addConstr(N[i,1,t] == N[i,1,t-1] + Sending[i,t-1])
                            
                for elem in S:
                    i = elem.id
                    
                    for t in range(1, Tc+1):
                        linkLength = getEuclideanDis(Node.idcase[Link.idcase[i].from_node].x, 
                                                     Node.idcase[Link.idcase[i].to_node].x, 
                                                     Node.idcase[Link.idcase[i].from_node].y, 
                                                     Node.idcase[Link.idcase[i].to_node].y) / 1000
                        
                        bwtt = t - int((linkLength / w) / clock * 3600)
                        fftt = t - int((linkLength / vf) / clock * 3600)
                        
                        if bwtt <= 0:
                            m.addConstr(Receiving[i,t-1] <= 0 + 120*K[i] * linkLength - N[i,0,t-1])
                            m.addConstr(Receiving[i,t-1] >= 0 + 120*K[i] * linkLength - N[i,0,t-1] - M * zx1[i,t])
                        else:
                            m.addConstr(Receiving[i,t-1] <= N[i,1,bwtt] + 120*K[i] * linkLength - N[i,0,t-1])
                            m.addConstr(Receiving[i,t-1] >= N[i,1,bwtt] + 120*K[i] * linkLength - N[i,0,t-1] - M * zx1[i,t])
                            
                        m.addConstr(Receiving[i,t-1] <= PQ[i])
                        m.addConstr(Receiving[i,t-1] >= PQ[i] - M * (1 - zx1[i,t]))
                        
                        m.addConstr(G[i,t-1] <= elem.flowinput * 8 / 3600 * clock * t - N[i,0,t-1])
                        m.addConstr(G[i,t-1] >= elem.flowinput * 8 / 3600 * clock * t - N[i,0,t-1] - M * zx2[i,t])
                        m.addConstr(G[i,t-1] <= Receiving[i,t-1])
                        m.addConstr(G[i,t-1] >= Receiving[i,t-1] - M * (1 - zx2[i,t]))
                        
                        m.addConstr(N[i,0,t] == N[i,0,t-1] + G[i,t-1])
                        
                        m.addConstr(space[i,1,t] == w/v * (Kjam[i] - n[i,1,t]))
                        if fftt <= 0:
                            m.addConstr(freeflow[i,t-1] == 0 - N[i,1,t-1]) 
                        else:
                            m.addConstr(freeflow[i,t-1] == N[i,0,fftt] - N[i,1,t-1])
                        m.addGenConstrMin(Sending[i,t-1], [freeflow[i,t-1], PQ[i], space[i,1,t]])

                        # if fftt <= 0:
                        #     m.addConstr(Sending[i,t-1] <= 0 - N[i,1,t-1])
                        #     m.addConstr(Sending[i,t-1] >= 0 - N[i,1,t-1] - M * zx3[i,t])
                        # else:
                        #     m.addConstr(Sending[i,t-1] <= N[i,0,fftt] - N[i,1,t-1])
                        #     m.addConstr(Sending[i,t-1] >= N[i,0,fftt] - N[i,1,t-1] - M *zx3[i,t])
                            
                        # m.addConstr(Sending[i,t-1] <= PQ[i])
                        # m.addConstr(Sending[i,t-1] >= PQ[i] - M * zx4[i,t])
                        
                        # m.addConstr(Sending[i,t-1] <= w/v * (Kjam[i] - n[i,1,t]))
                        # m.addConstr(Sending[i,t-1] >= w/v * (Kjam[i] - n[i,1,t]) - M * (2 - zx3[i,t] - zx4[i,t]))
                        m.addConstr(N[i,1,t] == N[i,1,t-1] + Sending[i,t-1])
                        m.addConstr(f[i,1,t] == Sending[i,t-1])
                        
                #flow into link in E
                for elem in E:
                    i1 = elem.id
                    # if Node.idcase[elem.to_node].type == 0:
                    #     continue
                    for t in range(1, Tc+1):
                        m.addConstr(f[i1,1,t] == quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
                        
                #last cell flow on link i in S
                for elem in S:
                    i = elem.id
                    for t in range(1, Tc+1):
                        m.addConstr(f[i,maxcell,t] == quicksum(fI[i,i1,t] for i1 in getExitLinkByApproachLink(i)))
                        
            # Integrate new variables
            #     m.update()
            ##link flow input
                # for elem in S:
                #     i = elem.id
                #     for t in range(1, Tc+1):
                #         # m.addConstr(f[i,1,t] <= Link.idcase[i].flowinput / 3600 * clock)
                #         # m.addConstr(f[i,1,t] >= Link.idcase[i].flowinput / 3600 * clock-z1[i,1,t]*M)
                        
                #         flow_input = random.choice(np.random.poisson(Link.idcase[i].flowinput / 3600 * clock, size=10))
                #         m.addConstr(f[i,1,t] <= flow_input)
                #         m.addConstr(f[i,1,t] >= flow_input-z1[i,1,t]*M)
                        
                #         m.addConstr(f[i,1,t] <= PQ[i])
                #         m.addConstr(f[i,1,t] <= w/v*(N[i]-n[i,1,t]))
                        
                #         m.addConstr(f[i,1,t] >= PQ[i]-z2[i,1,t]*M)
                #         m.addConstr(f[i,1,t] >= w/v*(N[i]-n[i,1,t])-(2-z1[i,1,t]-z2[i,1,t])*M)
            #normal link flow constrains, from cell 2 to the last cell
                # for i in U:
                #     for j in range(2,maxcell):   
                #         for t in range(1, Tc+1):
                #             m.addConstr(f[i,j,t] <= n[i,j-1,t])
                #             m.addConstr(f[i,j,t] <= PQ[i])
                #             m.addConstr(f[i,j,t] <= w/v*(N[i]-n[i,j,t]))
                #             m.addConstr(f[i,j,t] >= n[i,j-1,t]-z1[i,j,t]*M)
                #             m.addConstr(f[i,j,t] >= PQ[i]-z2[i,j,t]*M)
                #             m.addConstr(f[i,j,t] >= w/v*(N[i]-n[i,j,t])-(2-z1[i,j,t]-z2[i,j,t])*M)
                            
                #outflow of link i1 in E
                # for elem in E:
                #     i1 = elem.id
                #     for t in range(1, Tc+1):
                #         m.addConstr(f[i1,maxcell,t] == n[i1,maxcell-1,t])
            #CTM update
                for i in U:
                    for j in range(1,maxcell):
                        for t in range(1,Tc+1):
                            m.addConstr(n[i,j,t+1] == n[i,j,t] + f[i,j,t] - f[i,j+1,t])
                            
                # Lock Right turn movement with all green
                for key in movement_dict:
                    if movement_dict[key][1] == 3:
                        m.addConstr(phi[key[0], key[1]] == 1)
                            
                # Coordination
                for elem in E:
                    i1 = elem.id
                    for t in range(1, Tc+1):
                        m.addConstr(w/v*(Kjam[i1] - n[i1,1,t]) >= quicksum(fI[i,i1,t] for i in getApproachLinkByExitLink(i1)))
                            
                DeRT={}
                for i,i1 in movement:
                    for k in range(1,K[i]+1):
                        DeRT[i,i1,k]=m.addVar(vtype=GRB.BINARY,name='DeRT%s_%s_%s'%(i,i1,k))
                
            #movement flow constrains
                for i,i1 in movement:
                    for t in range(1, Tc+1):
                        m.addConstr(fI[i,i1,t]<=n[i,maxcell-1,t]*ratio[i,i1])
                        m.addConstr(fI[i,i1,t]<=PQ[i]*ratio[i,i1]*volve[i,i1,t])
                        m.addConstr(fI[i,i1,t]<=w/v*(Kjam[i]-n[i1,1,t]))
                        m.addConstr(fI[i,i1,t]>=n[i,maxcell-1,t]*ratio[i,i1]-z3[i,i1,t]*M)
                        m.addConstr(fI[i,i1,t]>=PQ[i]*ratio[i,i1]*volve[i,i1,t]-z4[i,i1,t]*M)
                        m.addConstr(fI[i,i1,t]>=w/v*(Kjam[i]-n[i1,1,t])-(2-z3[i,i1,t]-z4[i,i1,t])*M)
                        
            #1 lane permission
            #2. one lane can share multiple directions
            # constrain 28
                for elem in S:
                    i = elem.id
                    for k in range(1,K[i]+1):
                        m.addConstr(quicksum(DeRT[i,i1,k] for i1 in getExitLinkByApproachLink(i))>=1)
                        m.addConstr(quicksum(DeRT[i,i1,k] for i1 in getExitLinkByApproachLink(i))<=2)
                        
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
            #signal for right turn#
                # m.addConstr(phi[3,2]==1)
                # m.addConstr(phi[1,8]==1)
                # m.addConstr(phi[5,4]==1)
                # m.addConstr(phi[7,6]==1)
                # m.addConstr(sta[1,8]==0)
                # m.addConstr(sta[5,4]==0)
                # m.addConstr(sta[7,6]==0)
                # m.addConstr(sta[3,2]==0)
         
                for i,i1 in movement:
                    # m.addConstr(sta[i,i1]+phi[i,i1]<=1)
                    m.addConstr(sta[i,i1]+phi[i,i1]<=1+zs[i,i1]*M)
                    m.addConstr(1.01<=sta[i,i1]+phi[i,i1]+(1-zs[i,i1])*M)
                    for t in range(1, cc+1):
                        # constrain 14
                        m.addConstr(t >= c/clock * sta[i,i1] - zs[i,i1] * M - (1 - volve[i,i1,t]) * M)
                        m.addConstr(t <= c/clock * (sta[i,i1] + phi[i,i1]) + (1 - volve[i,i1,t])*M + zs[i,i1]*M)
                        
                        # constrain 15
                        m.addConstr(t >= c/clock * (sta[i,i1] + phi[i,i1]) - volve[i,i1,t]*M - zs[i,i1]*M - (1-zs1[i,i1,t]) * M)
                        m.addConstr(t <= c/clock * sta[i,i1] + volve[i,i1,t]* M + zs[i,i1]*M + zs1[i,i1,t] * M)
                        
                        # constrain 17
                        m.addConstr(t >= c/clock * (sta[i,i1] + phi[i,i1] - 1) - (1 - zs[i,i1]) * M - volve[i,i1,t] * M)
                        m.addConstr(t <= c/clock * sta[i,i1] + volve[i,i1,t] * M + (1 - zs[i,i1]) * M)
                        
                        # constrain 18
                        m.addConstr(t >= c/clock * sta[i,i1] - (1 - zs[i,i1])*M - (1 - volve[i,i1,t])*M - (1 - zs2[i,i1,t])*M)
                        m.addConstr(t <= c/clock * (sta[i,i1] + phi[i,i1] - 1) + (1 - volve[i,i1,t])*M + (1 - zs[i,i1])*M + zs2[i,i1,t]*M)
                        
            #order of movement   
                for i,i1,i2,i3 in conflict:
                    m.addConstr(omg[i,i1,i2,i3]+omg[i2,i3,i,i1]==1)
                    m.addConstr(sta[i,i1]+phi[i,i1]+gap/c <= sta[i2,i3]+omg[i,i1,i2,i3])
                    # for k1 in range(1, K[i]+1):
                    #     for k2 in range(1, K[i1]+1):
                    #         m.addConstr(sta[i,i1]+phi[i,i1]+gap/c <= sta[i2,i3]+omg[i,i1,i2,i3] + M*(2-DeRT[i,i1,k1]-DeRT[i2,i3,k2]))
        
            #make sure that the value of variables volve is the same in every cycle
                for i,i1 in movement:
                    for t in range(1,Tc+1-cc):
                        m.addConstr(volve[i,i1,t]==volve[i,i1,t+cc])
                        
            # total output
                OP={}
                for elem in E:
                    i = elem.id
                    for t in range(1,Tc+1):
                        OP=m.addVar(vtype=GRB.CONTINUOUS,name='OP%s_%s'%(i,t))
                        m.addConstr(OP==quicksum(f[i.id,maxcell,t] for i in E
                                                                    for t in range(1,Tc+1)))
                        
                D={}
                for i in U:
                    for t in range(1,Tc+1):
                        D[i,t]=m.addVar(vtype=GRB.CONTINUOUS,name='D%s_%s'%(i,t))
            
                for i in U:
                    for t in range(1,Tc+1):
                        m.addConstr(D[i,t]==clock*quicksum(n[i,j,t]-f[i,j+1,t] for j in range(1,maxcell)))
                DT={}
                for t in range(1,Tc+1):
                    DT[t]=m.addVar(vtype = GRB.CONTINUOUS,name='DT%s'%(t))
                for t in range(1,Tc+1):
                    m.addConstr(DT[t] == quicksum(D[i,t] for i in U))
                    
            
                for key in Movement.idcase:
                    if Movement.idcase[key].gdirect != 2:
                        continue
                    
                    if Movement.idcase[key].from_link not in S or Movement.idcase[key].to_link not in E:
                        continue
                    
                    from_link = Link.idcase[Movement.idcase[key].from_link]
                    fl_vector = [Node.idcase[from_link.to_node].x - Node.idcase[from_link.from_node].x, 
                             Node.idcase[from_link.to_node].y - Node.idcase[from_link.from_node].y]
                    
                    if fl_vector[1] != 0:
                        if abs(fl_vector[0] / fl_vector[1]) <= 0.333:
                            m.addConstr(sta[Movement.idcase[key].from_link, Movement.idcase[key].to_link] == 0)
                        
                m.update()
            #Set objective
                # m.setObjective(clock*quicksum(n[i.id,maxcell-1,t]-f[i.id,maxcell,t] for i in S
                #                                                                 for t in range(1,Tc+1)), GRB.MINIMIZE)
                
                # m.setObjective(clock*quicksum(-f[i.id,j,t] for i in S for j in range(1, maxcell)
                #                      for t in range(1,Tc+1)), GRB.MINIMIZE)
                
                m.Params.TimeLimit = 3600
                
                # m.Params.tuneResults = 1
                # m.Params.tuneTimeLimit = 7200
                # m.tune()
                # if m.tuneResultCount > 0:
                #     # Load the best tuned parameters into the model
                #     m.getTuneResult(0)
                #     # Write tuned parameters to a file
                #     m.write('tune_%s.prm' % str(ndkey))
                    
                m.Params.Heuristics = 0.5
                # m.Params.Method = 1
                
                m.setObjective(clock*quicksum(n[i.id,j,t] for i in S for j in range(1,maxcell)
                                                                                for t in range(1,Tc+1)), GRB.MINIMIZE)
                            
                m.optimize()
                m.write("model.lp")
                
                origin_obj = 0
                for i in S:
                    for t in range(1, Tc+1):
                        origin_obj += clock*(n[i.id, maxcell-1, t].x - f[i.id, maxcell, t].x)
                print(origin_obj)
                
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
                
                cvn_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
                for i in U:
                    for j in [0,1]:
                        cvn_df.loc[str(i)+', '+str(j)] = np.zeros(Tc)
                        for t in range(1, Tc+1):
                            cvn_df.loc[str(i)+', '+str(j)][t] = N[i,j,t].x
                cvn_df.to_csv("output_cvn.csv")
                            
                trans_flow_df = pd.DataFrame(columns=[i for i in range(1, Tc+1)])
                for i in U:
                    trans_flow_df.loc[str(i)] = np.zeros(Tc)
                    for t in range(1, Tc+1):
                        trans_flow_df.loc[str(i)][t] = G[i,t].x
                trans_flow_df.to_csv("output_transflow.csv")
                
                for key in sta:
                    theta_output[key] = sta[key].x
                    
                for key in phi:
                    phi_output[key] = phi[key].x
                    
                for key in STA:
                    theta_upper_output[key] = STA[key].x
                    
                for key in PHI:
                    phi_upper_output[key] = PHI[key].x
                    
                zxx = []
                for elem in S:
                    i = elem.id
                    for t in range(1, Tc+1):
                        zxx.append(zx3[i,t].x + zx4[i,t].x)
                            
                print(zxx)
                print(np.max(zxx))
        
        return (theta_output, phi_output, theta_upper_output, phi_upper_output)
        
    except AttributeError:
        print('Encountered an attribute error')
    
            
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
    # lp_optimize(200, 1, *optimization_distributed(200))
    output = optimization_distributed(200)
    # print("theta:", output[0], '\n')
    # print("phi:", output[1], '\n')
    # print("THETA:", output[2], '\n')
    # print("PHI:", output[3], '\n')
    # lp_optimize(200, 1)