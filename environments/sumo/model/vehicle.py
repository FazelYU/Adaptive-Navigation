"""Incoming data loaders

This file contains loader classes that allow reading iteratively through
vehicle entry data for various different data formats


Classes:
    Vehicle
    Entry

"""


class Vehicle():
    """Representation of a single vehicle."""
    def __init__(self, entry, pce):

        # vehicle properties
        self.id = entry.id
        self.type = entry.type

        # set pce multiplier based on type
        if self.type in ["passenger", "DEFAULT_VEHTYPE", "veh_passenger"]:
            self.multiplier = pce["car"]
        elif self.type in ["motorcycle", "veh_motorcycle"]:
            self.multiplier = pce["moto"]
        elif self.type in ["truck", "veh_truck"]:
            self.multiplier = pce["truck"]
        elif self.type in ["bus", "veh_bus"]:
            self.multiplier = pce["bus"]
        elif self.type in ["taxi", "veh_taxi"]:
            self.multiplier = pce["taxi"]
        else:
            self.multiplier = pce["other"]
            

        self.last_entry = None
        self.new_entry = entry


    def update(self):
        """Shift new entries to last and prepare for new"""

        if self.new_entry != None:
            self.last_entry = self.new_entry
            self.new_entry = None


    def distance_moved(self):
        """Calculate the distance the vehicle traveled within the same edge"""
        return self.new_entry.pos - self.last_entry.pos


    def approx_distance_moved(self, time_diff):
        """Approximate the distance the vehicle traveled between edges"""
        return self.new_entry.speed * time_diff


    def __repr__(self):
        return ('{}({})'.format(self.__class__.__name__, self.id))



class Entry():
    """Representation of a single timestep sensor entry of a vehicle."""
    def __init__(self, entry, time):

        # vehicle properties
        self.id = entry['id']
        self.type = entry['type']
        self.time = time

        # extract edge and lane ids
        self.edge_id = entry['lane'].rpartition('_')[0]
        self.lane_id = entry['lane'].rpartition('_')[1]

        # store position in edge
        self.pos = float(entry['pos'])
        self.speed = float(entry['speed'])
        
        # store location/speed data
        # self.x = float(entry['x'])
        # self.y = float(entry['y'])


    def __repr__(self):
        return ('{}({})'.format(self.__class__.__name__, self.id))
        