# Lane_based_model

## Overview
Lane_based_model is a traffic signal optimization model based on the Cell Transmission Model (CTM) and the Link Transmission Model (LTM). Gurobipy is used here as solver of the optimization model.
<br>Details of the optimization model will be shown later as the paper is accepted and published.

## Input and output
### Input of the model:
<b>link.csv: </b>All links in the network.
<br><b>node.csv:</b> All nodes, include intersections in the network. Node type 6 represents dead_ends and 0 represents intersection.
<br><b>movement_calibration.csv: </b></br> Movement_calibration.csv provide the program about traffic flow into intersections and turning ratio of each movements.

Where link.csv and node.csv have similar format with DTALite developed by Prof. Xuesong Zhou (https://github.com/xzhou99/Dtalite_traffic_assignment). 

This program can automaticlly generate movement and confilct in intersections as long as regular network defined in link.csv. That is, if you define a non-regular intersection, such as an intersection has 5 arms, you must define movement of the intersection yourself.

### Output of the model:
<b>output_density.csv:</b> Density of all cells in the network.
<br><b>output_flow.csv:</b> Traffic flow of all cells in the network.
<br><b>output_valve.csv:</b> Binary variables that represent signal of movements, 1 represents green and 0 represents red.
<br><b>output_signal.csv:</b> Traffic signal timing.
<br><b>output_movement_flow.csv:</b> Traffic flow of all movements in intersections.
<br><b>Cumulative_Vehicle_Number.csv:</b> Cumulative vehicle nubmer of all links.
<br><b>Lane_occupancy.csv:</b> Lane occupancy of all apporach link of intersections, based on different movement.

## Usage
### Optimization_CTM(c, isReturnVal=False, isOptimizeCycle=False)
The single intersection optimization model based on CTM. 
#### Parameters
<b>c:</b> Where c is cycle of the traffic signal. 
<br><b>isReturnVal:</b> This function will return objective value of the optimization model if isReturnVal is Ture, or signal timing result will be returned.
<br><b>isOptimizeCycle:</b> If isOptimizaCycle is Ture, c will be used as cycle, otherwise you may define cycle_dict in the program to provide specific cycle for every intersections, as line 897 shows.

#### Notes
One may focus on only an intersection to test the model. Line 905 and line 906 are example that add a condition to exlucde all intersections except interested intersection by its node id.

#### Returns
<b>If isReturnVal is True:</b> objective value of the optimization model.
<br><b>If isReturnVal is False:</b> Signal timing of all single intersections.

### lp_optimize(c, init_theta={}, init_phi={}, init_theta_upper={}, init_phi_upper={}, input_chromo=[], isReturn=False)
The coordinate network traffic signal optimization model based on CTM and LTM.
#### Parameters
<b>c:</b> Cycle of traffic signal. If cycle_dict is not defined. However, cycle_dict in the line 367 is strongly recommended to use.
<br><b>init_theta, init_phi, init_theta_upper, init_phi_upper:</b> Signal timing of all single intersections solved by CTM model which is used here as initial solution to improve efficiency of solving the model. If all of them are blank, the model will not use these as initial solutions, and solving the model may be very slow that a feasible solution may not be obtained in an hour.
<br><b>input_chromo: </b> Genetic algorithm is used here to find optimal cycle. However, as the model is hard to solve in short time, this is not recommended to use.

#### Notes
About cycle_dict, the key of cycle_dict is node id of the intersection and value is cycle of traffic signal.

### readNetwork()
### initMovement()
### calibrate_all()
These three functions will read link.csv and node.csv, initialize all movements, and calibrate input flow and turning ratios.
