from copy import deepcopy
from logging import root
from turtle import width
import matplotlib
import matplotlib.axes
from matplotlib.pylab import f
import numpy as np
from typing import Protocol, Union, Tuple, Optional, List, Dict
from abc import ABC, abstractmethod
from math import pi, sqrt, sin, cos, radians, degrees
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class Point:
    x: float
    y: float
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def to_array(self) -> np.ndarray:
        return np.array([[self.x], [self.y]])


@dataclass
class TransParams:
    angle_deg: Optional[float] = None
    pivot: Optional[Point] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    
    def __repr__(self):
        return f"TransParams(angle_deg={self.angle_deg}, pivot={self.pivot}, dx={self.dx}, dy={self.dy})"
    

def distance(p1: Point, p2: Point) -> float:
    """Computes the Euclidean distance between two points."""
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def get_2D_rotation_matrix(angle: float) -> np.ndarray:
    """Returns a 2D rotation matrix for a given angle in degrees."""
    angle_rad = radians(angle)
    return np.array([[cos(angle_rad), -sin(angle_rad)],
                     [sin(angle_rad), cos(angle_rad)]])
    
    
def rotate2D(point: Point, transform_params: TransParams) -> Point:
    """Rotate a point by a given angle in degrees around a pivot point."""
    pivot = transform_params.pivot if transform_params.pivot else Point(0, 0)
    angle_deg = transform_params.angle_deg if transform_params.angle_deg else 0
    rotation_matrix = get_2D_rotation_matrix(angle_deg)
    
    point = point.to_array() - pivot.to_array()
    point = rotation_matrix @ point
    point = point + pivot.to_array()
    
    return Point(point[0, 0], point[1, 0])


@dataclass
class RectangleParams:
    width: float
    height: float
    center: Point = field(default_factory=lambda: Point(0, 0))
    rotation: float = 0
   
        
class Rectangle:
    def __init__(self, *args, **kwargs):
        # Handle positional arguments: width, height, [center], [rotation]
        if len(args) >= 2:
            kwargs = {
                'width': args[0],
                'height': args[1],
                'center': args[2] if len(args) > 2 else Point(0, 0),
                'rotation': args[3] if len(args) > 3 else 0
            }

        params = RectangleParams(**kwargs)
        
        self.width = params.width
        self.height = params.height
        self._center = params.center
        self._rotation = params.rotation
        
    @property
    def center(self) -> Point:
        return self._center
    
    @center.setter
    def center(self, new_center: Union[Point, Tuple]) -> None:
        if isinstance(new_center, tuple):
            new_center = Point(*new_center)
        self._center = new_center
    
    def translate(self, dx: float, dy: float) -> 'Rectangle':
        self.center = Point(self.center.x + dx, self.center.y + dy)
        return self
        
    @property
    def rotation(self) -> float:
        return self._rotation
    
    @rotation.setter
    def rotation(self, angle_deg: float) -> None:
        self._rotation = angle_deg
        
    @property
    def area(self) -> float:
        return self.width * self.height
       
    def moment_inertia(self, angle_deg:float=0, pivot: Union[Point, Tuple]=None) -> Tuple:
        """Calculate the rotated and transferred moment of inertia about the given pivot"""
        angle_rad = radians(angle_deg)
        if isinstance(pivot, tuple):
            pivot = Point(*pivot)
        pivot = pivot or self.center
        
        # Centroidal moments of inertia
        ix = (self.width * self.height**3) / 12
        iy = (self.height * self.width**3) / 12
        
        # Rotated centroidal moments
        i_avg = (ix + iy) / 2
        i_diff = (ix - iy) / 2
        ix_r = i_avg + i_diff * cos(2 * angle_rad)
        iy_r = i_avg - i_diff * cos(2 * angle_rad)
        ixy_r = i_diff * sin(2 * angle_rad)
        
        # Parallel axis theorem for transfer to pivot
        dx = pivot.x - self.center.x
        dy = pivot.y - self.center.y
        ix_rt = ix_r + self.area * dy**2
        iy_rt = iy_r + self.area * dx**2
        ixy_rt = ixy_r + self.area * dx * dy
        
        return ix_rt, iy_rt, ixy_rt
    
    @property
    def corners(self) -> List[Point]:
        """Return four corners of the rectangle, counterclockwise from bottom-left"""
        dx, dy = self.width / 2, self.height / 2
        offsets = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        corners = [self.center + Point(x, y) for x, y in offsets]

        if self.rotation:
            trans = TransParams(angle_deg=self.rotation, pivot=self.center)
            corners = [rotate2D(corner, trans) for corner in corners]

        return corners

    def rotate(self, angle_deg: float, pivot: Optional[Union[Point, Tuple]] = None) -> 'Rectangle':
        trans_params = TransParams(angle_deg=angle_deg)
        if isinstance(pivot, tuple):
            pivot = Point(*pivot)
        trans_params.pivot = pivot if pivot else self.center 
        self.center = rotate2D(self.center, trans_params)
        self.rotation = self.rotation + angle_deg
    
    def plot(self, ax=None, show_center=True, **kwargs):
        corners = self.corners
        corners.append(corners[0]) # Add the first corner again to close the polygon
        corners = [(p.x, p.y) for p in corners] # Convert to tuples
        xs, ys = zip(*corners)
        
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('fig_size', (4, 4)))
            
        ax.set_aspect('equal')
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=kwargs.get('tick_size', 8))
        ax.plot(xs, ys, color=kwargs.get('line_color', 'black'), linewidth=kwargs.get('line_width', 1), zorder=5)
        ax.fill(xs, ys, color=kwargs.get('fill_color', 'blue'), alpha=kwargs.get('alpha', 0.5))
        if show_center:
            ax.plot(self.center.x, self.center.y, marker='o', color=kwargs.get('point_color', 'red'))
            
        return ax
    
    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height}, center={self.center}, rotation={self.rotation})"
    
    def __str__(self):
        return f"Rectangle(width={self.width:.3f}, height={self.height:.3f}, center={self.center}, rotation={self.rotation:.3f})"


@dataclass
class ISectionParams:
    top_flange: RectangleParams
    web: RectangleParams
    bottom_flange: Optional[RectangleParams] = None

    def __post_init__(self):
        if self.bottom_flange is None:
            self.bottom_flange = self.top_flange


class ISection:
    def __init__(self, params: ISectionParams) -> None:
        self._create_components(params)
        self._align_centroid_to_origin()
        
    def _create_components(self, params: ISectionParams) -> None:
        self._web: Rectangle = deepcopy(params.web)
        self._web.center = Point(0, 0)
        
        self._top_flange: Rectangle = deepcopy(params.top_flange)
        self._top_flange.center = Point(0, 0.5 * (params.web.height + params.top_flange.height))

        self._bottom_flange: Rectangle = deepcopy(params.bottom_flange)
        self._bottom_flange.center = Point(0, -0.5 * (params.web.height + params.bottom_flange.height))

    def _align_centroid_to_origin(self):
        centroid = self._get_centroid()
        for shape in self.components.values():
            shape.center = shape.center - centroid

    def _get_centroid(self):
        ax, ay = 0, 0
        for shape in self.components.values():
            ax += shape.area * (shape.center.x - self._web.center.x)
            ay += shape.area * (shape.center.y - self._web.center.y)
        return Point(ax / self.area, ay / self.area)
    
    @property
    def components(self) -> Dict[str, Rectangle]:
        return {"top_flange": self._top_flange, "bottom_flange": self._bottom_flange, "web": self._web}
    
    @property
    def corners(self) -> List[Point]:
        """Compute all corners of the I-section"""
        corners = [] 
        for shape in self.components.values():
            corners.extend(shape.corners)
        return corners
               
    @property
    def area(self) -> float:
        return self._web.area + self._bottom_flange.area + self._top_flange.area
    
    @property
    def center(self) -> Point:
        centroid = self._get_centroid()
        return self._web.center + centroid
    
    @center.setter
    def center(self, new_center: Union[Point, Tuple]) -> Point:
        if isinstance(new_center, tuple):
            new_center = Point(*new_center)
        for shape in self.components.values():
            shape.translate(new_center.x, new_center.y)
        return new_center
    
    def moment_inertia(self, angle_deg: float = 0, pivot: Optional[Union[Point, Tuple]] = None) -> Tuple:
        """Calculate the moment of inertia about the given pivot"""
        if isinstance(pivot, tuple):
            pivot = Point(*pivot)
        pivot = pivot or self.center
        
        ix, iy, ixy = 0, 0, 0
        for shape_name, shape in self.components.items():
            ix_r, iy_r, ixy_r = shape.moment_inertia(angle_deg= angle_deg, pivot=pivot)
            print(f"{shape_name}: ix: {ix_r:.3f}, iy: {iy_r:.3f}, ixy: {ixy_r:.3f}")
            ix += ix_r
            iy += iy_r
            ixy += ixy_r
        return ix, iy, ixy
    
    def translate(self, dx: float, dy: float) -> 'ISection':
        """Translate the I-section by dx and dy"""
        self.center = Point(self.center.x + dx, self.center.y + dy)
        return self
        
    def rotate(self, angle_deg: float, pivot: Optional[Union[Point, Tuple]] = None) -> 'ISection':
        """Rotate the I-section by the given angle in degrees around a pivot point"""
        flange_pivot = pivot if pivot else self.center
        self.components["web"].rotate(angle_deg, pivot)
        for shape_name, shape in self.components.items():
            if shape_name != "web":
                shape.rotate(angle_deg, flange_pivot)
        return self
     
    def plot(self, ax=None, **kwargs):
        figsize=kwargs.get('fig_size', (4, 4))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            plt.grid(True)
        
        for shape_name, shape in self.components.items():
            ax = shape.plot(ax=ax, show_center=False, **kwargs)
        center = self.center
        ax.plot([center.x], [center.y], marker='o', color=kwargs.get('point_color', 'red'), markersize=kwargs.get('markersize', 5), zorder=10)
        ax.set_aspect('equal')
        
        return ax