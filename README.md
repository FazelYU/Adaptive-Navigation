Paper: to be submitted

Abstract
----------------------------------------
Traffic congestion in urban road networks is a condition characterised by slower speeds, longer trip times, increased air pollution, and driver frustration. Traffic congestion can be attributed to a volume of traffic that generates demand for space greater than the available street capacity. A number of other specific circumstances can also cause or aggravate congestion, including traffic incidents, road maintenance work and bad weather conditions.


While construction of new road infrastructure is an expensive solution, traffic flow optimization using route planning algorithms is considered a more economical and sustainable alternative. Currently, well-known publicly available car navigation services, such as Google Maps and Waze, help people with route planning. These systems mainly rely on variants of the popular Shortest Path First (SPF) algorithm to suggest a route, assuming a static network. However, road network conditions are dynamic, rendering the SPF route planning algorithms to perform sub-optimally at times. In addition, SPF is a greedy algorithm. So, while it can yield locally optimal solutions that approximate a globally optimal solution in a reasonable amount of time, it does not always produce an optimal solution. For example, in a limited road capacity, the SPF routing algorithm can cause congestion by greedily routing all vehicles to the same road (towards the shortest path). 


To address limitations and challenges of the current approach to solve the traffic congestion problem, we propose **_a network-aware multi-agent reinforcement learning_ (MARL)** model for the navigation of a fleet of vehicles in the road network. The proposed model is _adaptive_ to the current traffic conditions of the road network. The main idea is that a Reinforcement Learning (RL) agent is assigned to every road intersection and operates as a _**router agent**_, responsible for providing routing instructions to a vehicle in the network. The vehicle traveling in the road network is aware of its final destination but not its exact full route/path to it. When it reaches an intersection, it generates _**a routing query**_ to the RL agent assigned to that intersection, consisting of its final destination. The RL agent generates _**a routing response**_ based on (i) the vehicle’s destination, (ii) the current state of the network in the neighborhood of the agent aggregated with a shared graph attention network (GAT) model, and (iii) routing policies learned by cooperating with other RL agents assigned to neighboring intersections. The vehicle follows the routing response from the router agents until it reaches its destination. 


Through an extensive experimental evaluation on both synthetic and realistic road networks we demonstrate that the proposed MARL model can outperform the SPF algorithm by (up to) 20\% in average travel time.

Example:
-----------------------------
The figure below gives an example of routing of two vehicles with two different destinations using the MARL model:

![alt text](https://github.com/FazelYU/Adaptive-Navigation/blob/add-license-1/Saved_Results/MARL%20Model%20Example.gif)

Model Architecure:
-----------------------------
The figure below shows the archeticture of the Adaptive Navigation algorithm:
![alt text](https://github.com/FazelYU/Adaptive-Navigation/blob/add-license-1/Saved_Results/Arch.png)

Reslts:
-----------------------------
The figures below show two test cases, the 5x6 grid network, and the abstracted network of DT Toronto extracted from <a href="https://www.openstreetmap.org/relation/2989349#map=13/43.6470/-79.3794"> Open Streets Map </a>:
<table>
  <tr>
    <td><img src="/Saved_Results/Networks/Toronto_Abstracted.png" width=500 height=400> <figcaption>Fig.1.1 - DT Toronto</td>
    <td><img src="/Saved_Results/Networks/net5x6.png" width=500 height=400> <figcaption>Fig.1.2 - 5x6 Grid.</figcaption></td>
  </tr>
<!--   <tr>
    <td><img src="/Saved_Results/Networks/Toronto_Abstracted.png" width=270 height=480></td>
  </tr> -->
</table>
The figures below show the training results of the two test cases. Q-routing is a deep learning implementaion of <a href="https://proceedings.neurips.cc/paper/1993/file/4ea06fbc83cdd0a06020c35d50e1e89a-Paper.pdf"> Boyan, and Litman</a> IP network routing algorithm. The number of GAT layers in the Adaptive Navigation algorithm is a hyper parameter denoted as "h".

<table>
  <tr>
        <td><img src="/Saved_Results/AVTT/Toronto Average Travel Time.png" width=300 height=200> <figcaption>Fig.2.1 - Toronto Average Travel Time </figcaption></td>
    <td><img src="/Saved_Results/AVTT/Toronto Last 100 Episodes.png" width=300 height=200> <figcaption>Fig.2.2 - Toronto Average Travel Time Zoom</figcaption</td>
    <td><img src="/Saved_Results/AVTT/5x6 Average Travel Time.png" width=300 height=200> <figcaption>Fig.2.3 - 5x6 Average Travel Time</td>
    <td><img src="/Saved_Results/AVTT/5x6 Last 100 Episodes.png" width=300 height=200> <figcaption>Fig.2.4 - 5x6 Average Travel Time Zoom</figcaption></td>

  </tr>
</table>

 After the training is complete, for testing purposes, we generate 2000 uniformly distributed trips in each test case. We also consider two sensible baselines: 1- SPF (travel time shortest path first), and SPFWR (travel time shortest path first wih rerouting). Note that SPFWR is prohibitively computationally expensive as it needs to recompute all the shortest pathes in every time step. Table below compares the results of the testing phase:
  
  <table>
    <tr>
      <td></td>
      <td> DT Toronto </td>
      <td> 5x6 Grid </td>
    </tr>
    <tr>
      <td>AN (h=2)</td>
      <td> 479.3 </td>
      <td> 145.4 </td>
    </tr>
    <tr>
      <td>AN (h=1)</td>
      <td> <u> 476.4 </u> </td>
      <td> <b> 138.4 </b> </td>
    </tr>
    <tr>
      <td>AN (h=0)</td>
      <td> 477.6 </td>
      <td> <u> 143.7 </u> </td>
    </tr>
    <tr>
      <td>Q-routing</td>
      <td> inf (loop) </td>
      <td> 159.6 </td>
    </tr>
    <tr>
      <td>SPF</td>
      <td>551.7</td>
      <td> 173.4 </td>
    </tr>
    <tr>
      <td>SPFWR</td>
      <td> <b> 475.6 </b> </td>
      <td> 205.1 </td>
    </tr>
  </table>
In DT Toronto AN(h=1) has been able to perform as good as the SPFWR. 
  
On the other hand, counter-intutively, in the 5x6 grid network the SPFWR performs poorly. The reason for this observation can be rooted to the low capacity of the network. SPFWR greedily sends all the vehicles through the current shortest path and easily congests it. Moreover, SPFWR does not consider the waiting times in the queue of the traffic lights. As a result it may lead to unnecessary changes of the route that stuck further behind the traffic lights. In 4x5 network, AN(h=1) out performs other baselines.
  
The figure below, shows part of the simulation for the SPWR. One can see that the traffic in the network is increasing as the time pass by. Also, you can see that at times some roadsegments get overpopulated by the SPFWR algorithm.
  
![alt text](https://github.com/FazelYU/Adaptive-Navigation/blob/add-license-1/Saved_Results/whyTTSPfails.gif)

  
How to run:
-----------------------------
* to set the routing algorithm, edit line 47 of the run_exp.py. 
    ```
    line 47: config.routing_mode=routing_modes[1]
    ```

  sets the routing for AN(h=1)

* to change between training/testing, edit line 44 of the run_exp.py. e.g. for testing phase:
  ```
    line 44: config.training_mode=False
  ```
* to run toronto experiement, edit line 61 of run_exp.py to :
  ```
     line 61: network_name="toronto"
  ```
  and run:
  
```
  python run_exp.py
  
```
  
* to run 5x6 experiement, set line 61 of run_exp.py to : 
  
  ```
  line 61: network_name="5x6"
  
  ```
  and run:
  
```
  python run_exp.py  
```
  

