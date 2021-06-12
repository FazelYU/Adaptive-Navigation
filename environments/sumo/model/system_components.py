class System():
    """Generic MOE calculation object for any system."""

    def __init__(self, _id, edges, min_speed = 1):

        self.id = _id
        self.edges = edges
        self.min_speed = min_speed
        
        # setup vehicle counters
        self.v_current = 0          # vehicles in edge right now
        self.v_visited = 0          # total vehicles that passed through edge
        self.total_dist = 0         # total distance moved by vehicles in edge
        self.total_ideal_time = 0   # total free-flow time for same distance


    def update_entered(self, vehicle, time_diff, flow_speed):
        """Check if new vehicle entered, update counters appropriately."""


        # check if vehicle came in from a different system or outside
        if (vehicle.last_entry is None
            or vehicle.last_entry.edge_id not in self.edges):
            
            # update counters
            self.v_current += 1 * vehicle.multiplier
            self.v_visited += 1 * vehicle.multiplier
            distance = vehicle.approx_distance_moved(time_diff)

        # vehicle moved within the same system
        else:
            distance = vehicle.distance_moved()

        # correct total distance and ideal time for stopped cars by
        # enforcing minimum speed (default 1 m/s) per car
        distance = max(distance, self.min_speed * time_diff)
        distance *= vehicle.multiplier

        self.total_dist += distance
        self.total_ideal_time += distance / flow_speed


    def update_left(self, vehicle):
        """Check if vehicle just left and update counters appropriately."""

        # check if vehicle left to a different system or outside
        if (vehicle.new_entry is None
            or vehicle.new_entry.edge_id not in self.edges):
            
            # update counter
            self.v_current -= 1 * vehicle.multiplier


    def compute_metrics(self, time_diff):
        """Compute and update HCM MOE values for a generic system."""

        throughput = self.v_current / time_diff

        # if cars actually  passed
        if self.v_current != 0 and self.total_ideal_time != 0:


            # Total Delay calculation, min speed is 1 m/s to prevent artifacts
            total_time = time_diff * self.v_current
            total_delay = max(0, total_time - self.total_ideal_time)

            # Percent Incomplete Trips, Delay per Trip, Travel Time Index
            pit = self.v_current / self.v_visited
            dpt = total_delay / self.v_current

            if self.total_ideal_time == 0:
                print(self.total_ideal_time)
                print(self)

            tti = total_time / self.total_ideal_time

        # no cars passed, default values
        else:

            total_delay = 0
            pit = 0 
            dpt = 0
            tti = 1

        # reset counters
        self.total_dist = 0
        self.total_ideal_time = 0

        self.metrics = {"pit": pit, "thr": throughput,
            "td": total_delay, "dpt": dpt, "tti": tti}

        return self.metrics


    def __repr__(self):
        return ('{}({})'.format(self.__class__.__name__, self.id))



class EdgeSystem(System):
    """MOE calculation object for a single road network edge."""
    
    def __init__(self, edge):
        super().__init__(edge.id, [edge])

        self.edge = edge


    def update_entered(self, vehicle, time_diff):
        """Update total ideal time specifically for single edge systems"""

        super().update_entered(vehicle, time_diff, self.edge.flow_speed)



class MultiEdgeSystem(System):
    """Generic MOE calculation object for any system with multiple edges."""

    def __init__(self, _id, edge_systems):
        self.edge_systems = {system.edge.id:system for system in edge_systems}
        super().__init__(_id, set(self.edge_systems.keys()))


    def update_entered(self, vehicle, time_diff):
        """Update total ideal time specifically for multi-edge systems"""

        # get the flow speed of the edge the vehicle is currently in
        super().update_entered(vehicle, time_diff, 
            self.edge_systems[vehicle.new_entry.edge_id].edge.flow_speed)


class PathSystem(MultiEdgeSystem):
    """MOE calculation object for a single road network path system."""
    
    def __init__(self, _id, edge_systems):
        super().__init__(_id, edge_systems)

        self.name = _id
        

class CustomSystem(MultiEdgeSystem):
    """MOE calculation object for user-created multi-edge systems."""
    
    def __init__(self, _id, edge_systems, name):
        super().__init__(_id, edge_systems)

        self.name = name
        

    def __repr__(self):
        return ('{}({}|{})'.format(self.__class__.__name__,self.name,self.id))