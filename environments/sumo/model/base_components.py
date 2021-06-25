import re, copy

class Junction():
    """Representation of a junction in a road network."""
    def __init__(self, junction):

        self.id = junction['id']
        self.internal_edges = {lane.rpartition('_')[0]
            for lane in junction['intLanes'].split()}

    def __repr__(self):
        return ('{}({})'.format(self.__class__.__name__, self.id))

class Connection():
    id = 0
    def __init__(self, fromEdge, toEdge):
        self.fromEdge = fromEdge
        self.toEdge = toEdge
        self.id = Connection.id
        Connection.id += 1

class Edge():
    """Representation of an edge in a road network."""
    def __init__(self, edge, lanes):

        self.id = edge['id']

        # if normal function, store normal properties
        if 'function' not in edge or edge['function'] == "normal":

            self.function = "normal"            
            # self.type = edge['type']
            self.type = 'normal'
            self.from_id = edge['from']
            self.to_id = edge['to']

        # internal/junction edge, no properties
        else:
            self.function = edge['function']
            self.type = "internal"
            self.from_id = None
            self.to_id = None

        # get nr. of lanes and edge length (use middle lane)
        self.lanes =[Lane(self.id, lane) for lane in lanes]
        self.lane_count = len(self.lanes)
        self.length = self.lanes[round(len(self.lanes)/2)-1].length

        # if edge has attribute 'shape', use it, otherwise get it from lanes
        if "shape" in edge:
            self.shape = Shape(edge['shape'])
        else:
            self.shape = self.lanes[round(len(self.lanes)/2)-1].shape

        # calculate edge min speed
        self.flow_speed = min([lane.speed for lane in self.lanes])


    def __repr__(self):
        return ('{}({})'.format(self.__class__.__name__, self.id))

class Lane():
    """Representation of a single lane of an edge in a road network."""
    def __init__(self, edge_id, lane):

        self.id = lane['id']
        self.edge = edge_id
        self.index = lane['index']
        self.speed = float(lane['speed'])
        self.length = float(lane['length'])
        self.shape = Shape(lane['shape'])
        # self.disallow = lane['disallow'].split('_')

class Shape():
    """Representation of the linear shape of a lane"""

    def __init__(self, serialized):
        """Deserialize a shape string into coordinate pairs"""
        coord_pairs = [pair.split(',') for pair in serialized.split(' ')]
        self.points = [(float(x), float(y)) for x, y in tuple(coord_pairs)]


    def transform(self, old, new):
        """Transform the shape points from one rectangle space to another."""

        # create a copy of this Shape and loop through its points
        new_shape = copy.deepcopy(self)
        for index, (x, y) in enumerate(self.points):

            # project points
            new_x = (x-old[0]) * (new[2]-new[0]) / (old[2]-old[0]) + new[0]
            new_y = (y-old[1]) * (new[3]-new[1]) / (old[3]-old[1]) + new[1]
            new_shape.points[index] = (new_x, new_y)

        return new_shape


    def get_center(self, rounded=False):
        """Calculate and return the center point of this lane."""

        x, y = zip(*[pair for pair in self.points])
        
        if rounded:
            return round(sum(x)/len(x)), round(sum(y) / len(y))
        else:
            return sum(x)/len(x), sum(y) / len(y)


    def __str__(self):
        """Serialize and return a shape string from the coordinate pairs"""
        coord_pairs = [str(x) + ',' + str(y) for x, y in self.points]
        return ' '.join(coord_pairs)