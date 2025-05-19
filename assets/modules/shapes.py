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
import pandas as pd
import os
from functools import lru_cache
from itertools import combinations
import shapely.geometry as sg 

@dataclass
class Point:
    x: float
    y: float
    
    def __add__(self, other: Union['Point', Tuple[float, float]]) -> 'Point':
        if isinstance(other, tuple):
            return Point(self.x + other[0], self.y + other[1])
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Union['Point', Tuple[float, float]]) -> 'Point':
        if isinstance(other, tuple):
            return Point(self.x + other[0], self.y + other[1])
        return Point(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def to_array(self) -> np.ndarray:
        return np.array([[self.x], [self.y]])
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class TransParams:
    angle_deg: Optional[float] = None
    pivot: Optional[Point] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    
    def __repr__(self):
        return f"TransParams(angle_deg={self.angle_deg}, pivot={self.pivot}, dx={self.dx}, dy={self.dy})"

class Circle:
    def __init__(self, radius: float, center: Union[Point, Tuple]= Point(0, 0)) -> None:
        self.radius = radius
        if isinstance(center, tuple):
            self._center = Point(*center)
        elif isinstance(center, Point):
            self._center = center
        else:
            raise TypeError("center must be a tuple (x, y) or a Point object.")
    
    @property
    def area(self) -> float:
        return np.pi * self.radius**2
    
    @property
    def center(self) -> Point:
        return self._center
    
    @center.setter
    def center(self, new_center: Union[Point, Tuple]) -> None:
        if isinstance(new_center, tuple):
            self._center = Point(*new_center)
        elif isinstance(new_center, Point):
            self._center = new_center
        else:
            raise TypeError("center must be a tuple (x, y) or a Point object.")
        
    def plot(self, ax: matplotlib.axes.Axes = None, **kwargs) -> matplotlib.axes.Axes:
        
        if ax is None:
            fig_size = kwargs.get('fig_size')
            _, ax = plt.subplots(figsize=kwargs.get('fig_size', fig_size))
            
        circle = plt.Circle(
            (self.center.x, self.center.y), 
            self.radius,
            fill=False if kwargs.get('facecolor') is None else True,
            facecolor=kwargs.get('facecolor'), 
            edgecolor=kwargs.get('edgecolor'),
            linewidth=kwargs.get('linewidth'),
            alpha=kwargs.get('alpha'), 
            zorder=kwargs.get('zorder'),
        )
        
        ax.add_artist(circle)
        ax.set_aspect('equal')
        ax.grid(True)
        
        if kwargs.get('centeroid_color') is not None:
            ax.plot(self.center.x, self.center.y, marker='o', color=kwargs.get('centeroid_color'))

        return ax
    
    def __repr__(self):
        return f"Circle(radius={self.radius}, center={self.center})"
    
    def __str__(self):
        return f"Circle(radius={self.radius:.3f}, center={self.center})"

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

        center = kwargs.get('center', Point(0, 0))
        if isinstance(center, tuple):
            center = Point(*center)
        kwargs['center'] = center

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
    
    def plot(self, ax: matplotlib.axes.Axes = None, **kwargs):
        corners = self.corners
        corners.append(corners[0]) # Add the first corner again to close the polygon
        corners = [(p.x, p.y) for p in corners] # Convert to tuples
        xs, ys = zip(*corners)
        
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize'))
            
        ax.set_aspect('equal')
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=kwargs.get('ticksize'))
        ax.plot(xs, ys, color=kwargs.get('edgecolor', 'black'), linewidth=kwargs.get('linewidth', 1), zorder=kwargs.get('zorder'))
        
        if kwargs.get('facecolor') is not None:
            ax.fill(xs, ys, color=kwargs.get('facecolor'), alpha=kwargs.get('alpha', 0.5), zorder=kwargs.get('zorder'))
            
        if kwargs.get('centroid_color') is not None:
            ax.plot(self.center.x, self.center.y, marker='+', color=kwargs.get('centroid_color'))
            
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
    

@dataclass
class AISCShape:
    label: str
    
    def props(self, as_dataframe: bool = False) -> Dict[str, float] | pd.DataFrame:
        """Return the properties of the AISC shape as dict or DataFrame"""
        series = get_aisc_shape(self.label)
        if as_dataframe:
            df = series.to_frame().T
            df.set_index("AISC_Manual_Label", inplace=True)
            return df
        
        return series.to_dict()
    
    @property
    def A(self) -> float:
        """Return the area of the AISC shape [in2]"""
        return self.props()['A']
    
    @property
    def Ix(self) -> float:
        """Return the moment of inertia about the x-axis [in4]"""
        return self.props()['Ix']
    
    @property
    def Iy(self) -> float:
        """Return the moment of inertia about the y-axis [in4]"""
        return self.props()['Iy']
    

@dataclass
class AISC_WSection(AISCShape):
    # Get label and raise error if it is of self.props()['Type'] == 'W'
    def __post_init__(self):
        if self.props()['Type'] != 'W':
            raise ValueError(f"AISC shape {self.label} is not a W-section.")
    
    @property
    def d(self) -> float:
        """Return the depth of the W-section [in]"""
        return self.props()['d']
    
    @property
    def bf(self) -> float:
        """Return the flange width of the W-section [in]"""
        return self.props()['bf']
    
    @property
    def tf(self) -> float:
        """Return the flange thickness of the W-section [in]"""
        return self.props()['tf']
    
    @property
    def tw(self) -> float:
        """Return the web thickness of the W-section [in]"""
        return self.props()['tw']
    
    @property
    def Awy(self) -> float:
        """Return the web area of the W-section [in2]"""
        return self.d * self.tw
    
    @property
    def alphaY(self) -> float:
        """shear shape factor along y-axis
        Return the ratio of web area (Aw) to the total area (A)
        """
        return self.Awy / self.A
    
    
# Utility functions =====================================================
# Load the full AISC database once and cache it
@lru_cache(maxsize=1)
def load_aisc_database():
    path = os.path.join('..', '..', 'assets', 'data', 'structure', 'aisc-shapes-database-v16.0.xlsx')
    return pd.read_excel(path, sheet_name='Database v16.0')


# Lookup shape properties from cached DataFrame
@lru_cache(maxsize=None)  # Cache each shape label lookup
def get_aisc_shape(label: str) -> pd.Series:
    df = load_aisc_database()
    result = df[df['AISC_Manual_Label'] == label]
    if result.empty:
        raise ValueError(f"Shape {label} not found in AISC shapes database.")
    return result.iloc[0]


def distance(p1: Point, p2: Point) -> float:
    """Computes the Euclidean distance between two points."""
    diff = p1 - p2
    return np.linalg.norm([diff.x, diff.y])


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


def circles_overlap(cricle1: Circle, circle2: Circle) -> bool:
    """
    Returns:
        bool: True if the circles overlap, False otherwise.
    """
    # center_dist = distance(c1.center, c2.center)
    # return center_dist < (c1.radius + c2.radius)
    c1 = sg.Point(cricle1.center.x, cricle1.center.y).buffer(cricle1.radius)
    c2 = sg.Point(circle2.center.x, circle2.center.y).buffer(circle2.radius)
    return c1.intersects(c2)


def no_circles_overlap(circles: list[Circle]) -> bool:
    """
    Returns:
        bool: True if all circles are disjoint (no overlaps), False otherwise.
    """
    for c1, c2 in combinations(circles, 2):
        if circles_overlap(c1, c2):
            return False
    return True


def circle_within_rectangle(circle: Circle, rectangle: Rectangle) -> bool:
    """
    Check if a circle is completely within a rectangle.
    
    Args:
        circle (Circle): The circle to check.
        rectangle (Rectangle): The rectangle to check against.
        
    Returns:
        bool: True if the circle is within the rectangle, False otherwise.
    """
    # bottum_left_corner = rectangle.corners[0]
    # top_right_corner = rectangle.corners[2]
    # circle_left = circle.center.x - circle.radius
    # circle_right = circle.center.x + circle.radius
    # circle_bottom = circle.center.y - circle.radius
    # circle_top = circle.center.y + circle.radius
    
    # return (
    #     circle_left >= bottum_left_corner.x and
    #     circle_right <= top_right_corner.x and
    #     circle_bottom >= bottum_left_corner.y and
    #     circle_top <= top_right_corner.y
    # )
    c = sg.Point(circle.center.x, circle.center.y).buffer(circle.radius)
    r = sg.box(rectangle.corners[0].x, rectangle.corners[0].y, rectangle.corners[2].x, rectangle.corners[2].y)
    return c.within(r)


def get_rectangle_net_area(rectangles: List[Rectangle], circles: List[Circle], show_plot: bool = True) -> float:
    """Compute the net area of rectangles after subtracting the area of the overlapping circle.

    Args:
        rectangles List(Rectangle): List of Rectangle objects.
        circles List(Circle): List of Circle objects to be subtracted from the rectangle.

    Returns:
        float: Net area of the rectangles after subtracting the area of the overlapping circle.
    """
    if not all(isinstance(rect, Rectangle) for rect in rectangles):
        raise TypeError("rectangles must be a list of Rectangle objects.")
    
    if not all(isinstance(circle, Circle) for circle in circles):
        raise TypeError("circles must be a list of Circle objects.")
    
    circle_geoms = [
        sg.Point(circle.center.x, circle.center.y).buffer(circle.radius)
        for circle in circles
    ]
    
    net_areas = []
    
    fig, ax = plt.subplots() if show_plot else (None, None)

    for rect in rectangles:
        r = sg.box(rect.corners[0].x, rect.corners[0].y, rect.corners[2].x, rect.corners[2].y)
        net_area = r.area
        
        for c in circle_geoms:
            net_area -= r.intersection(c).area
            
        net_areas.append(net_area)

        if show_plot:
            # Plot rectangle
            x, y = r.exterior.xy
            ax.fill(x, y, edgecolor='black', fill=False)
            
    if show_plot:
        for c in circle_geoms:
            x, y = c.exterior.xy
            ax.fill(x, y, color='red', alpha=0.3)

        ax.set_aspect('equal')
        ax.set_title("Rectangles and Rebars (Circles)")
        plt.show()
        
    return np.array([net_areas])
    
    
# Main function =========================================================
def main():
    pass
    
    
if __name__ == "__main__":
    main()