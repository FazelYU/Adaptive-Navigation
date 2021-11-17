Paper: under review Web 2022

Abstract
----------------------------------------
Traffic congestion in urban road networks is a condition characterised by slower speeds, longer trip times, increased air pollution, and driver frustration. Traffic congestion can be attributed to a volume of traffic that generates demand for space greater than the available street capacity. A number of other specific circumstances can also cause or aggravate congestion, including traffic incidents, road maintenance work and bad weather conditions.


While construction of new road infrastructure is an expensive solution, traffic flow optimization using route planning algorithms is considered a more economical and sustainable alternative. Currently, well-known publicly available car navigation services, such as Google Maps and Waze, help people with route planning. These systems mainly rely on variants of the popular Shortest Path First (SPF) algorithm to suggest a route, assuming a static network. However, road network conditions are dynamic, rendering the SPF route planning algorithms to perform sub-optimally at times. In addition, SPF is a greedy algorithm. So, while it can yield locally optimal solutions that approximate a globally optimal solution in a reasonable amount of time, it does not always produce an optimal solution. For example, in a limited road capacity, the SPF routing algorithm can cause congestion by greedily routing all vehicles to the same road (towards the shortest path). 


To address limitations and challenges of the current approach to solve the traffic congestion problem, we propose **_a network-aware multi-agent reinforcement learning_ (MARL)** model for the navigation of a fleet of vehicles in the road network. The proposed model is _adaptive_ to the current traffic conditions of the road network. The main idea is that a Reinforcement Learning (RL) agent is assigned to every road intersection and operates as a _**router agent**_, responsible for providing routing instructions to a vehicle in the network. The vehicle traveling in the road network is aware of its final destination but not its exact full route/path to it. When it reaches an intersection, it generates _**a routing query**_ to the RL agent assigned to that intersection, consisting of its final destination. The RL agent generates _**a routing response**_ based on (i) the vehicle’s destination, (ii) the current state of the network in the neighborhood of the agent aggregated with a shared graph attention network (GAT) model, and (iii) routing policies learned by cooperating with other RL agents assigned to neighboring intersections. The vehicle follows the routing response from the router agents until it reaches its destination. 


Through an extensive experimental evaluation on both synthetic and realistic road networks we demonstrate that the proposed MARL model can outperform the SPF algorithm by (up to) 20\% in average travel time.

Example:
-----------------------------
The figure below gives an example of routing of two vehicles with two different destinations using the MARL model.

![alt text](https://github.com/FazelYU/Adaptive-Navigation/blob/add-license-1/Saved_Results/MARL%20Model%20Example.gif)

Code
-----------------------------
prerequisites:
-----------------------------
How to run:
-----------------------------

