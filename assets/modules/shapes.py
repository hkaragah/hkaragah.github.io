##########################################
# Shapes Module
# Written by: Hossein Karagah
# Date: 2025-10-29
# Description: This module provides classes and functions to handle geometric shapes like circles, rectangles, and I-sections.
##########################################
from __future__ import annotations

from pathlib import Path
import sys

root = Path(__file__).resolve()
sys.path.extend([
    str(root.parents[1]),  # ../..
    str(root.parents[2]),  # ../../..
])

from copy import deepcopy
import matplotlib
import matplotlib.axes
import numpy as np
from math import pi, atan2
from typing import Protocol, Union, Tuple, Optional, List, Dict
from abc import ABC, abstractmethod
from math import pi, sqrt, sin, cos, radians, degrees
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import pandas as pd
import os
from functools import lru_cache
from itertools import combinations
from shapely import geometry as geom
from scipy.optimize import fsolve


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
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate the Euclidean distance to another point."""
        return np.linalg.norm(self.to_array() - other.to_array())
    
    def within(self, shape: Union['Rectangle', 'Circle']) -> bool:
        """
            Check if the point is within a rectangle or a circle.

            Args:
                shape (Rectangle or Circle): The shape to check against.

            Returns:
                bool: True if the point is within the shape, False otherwise.
        """
        p = geom.Point(self.x, self.y)
        if isinstance(shape, Rectangle):
            r = geom.box(
                shape.corners[0].x, 
                shape.corners[0].y, 
                shape.corners[2].x, 
                shape.corners[2].y
            )
            return p.within(r) or p.touches(r)
        
        elif isinstance(shape, Circle):
            center = shape.center
            distance_sq = (self.x - center.x) ** 2 + (self.y - center.y) ** 2
            return distance_sq <= shape.radius ** 2

        else:
            raise TypeError("Argument must be a Rectangle or Circle.")            
    
    def to_array(self) -> np.ndarray:
        return np.array([[self.x], [self.y]])
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

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
        self._radius = radius
        if isinstance(center, tuple):
            self._center = Point(*center)
        elif isinstance(center, Point):
            self._center = center
        else:
            raise TypeError("center must be a tuple (x, y) or a Point object.")
    
    @property
    def radius(self) -> float:
        return self._radius

    @property
    def area(self) -> float:
        return pi * self.radius ** 2
    
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
        
    def translate(self, translation: Tuple[float, float]) -> 'Circle':
        """Translate the circle by the given vector (dx, dy).
        This method moves the circle to a new position by adding the translation vector to its current center.
        
        Args:
            translation (Tuple[float, float]): The translation vector (dx, dy).
        Returns:
            Circle: The translated circle.
        """
            
        if isinstance(translation, tuple):
            dx, dy = translation
        else:
            raise TypeError("translation must be a tuple (dx, dy).")
        
        self._center = Point(self._center.x + dx, self._center.y + dy)
        return self
    
    def rotate(self, angle_deg: float, pivot: Optional[Union[Point, Tuple]] = None) -> 'Circle':
        """Rotate the circle by the given angle in degrees around a pivot point.
        
        Args:
            angle_deg (float): The rotation angle in degrees.
            pivot (Optional[Union[Point, Tuple]]): The pivot point around which the circle is rotated. If None, the circle is rotated around its own center.

        Returns:
            Circle: The rotated circle.
        """
        trans_params = TransParams(angle_deg=angle_deg)
        if isinstance(pivot, tuple):
            pivot = Point(*pivot)
        trans_params.pivot = pivot if pivot else self.center 
        self._center = rotate2D(self._center, trans_params)
        return self
    
    def overlapped_with(self, other_circle: 'Circle') -> bool:
        """
        Returns:
            bool: True if the circles overlap, False otherwise.
        """
        # center_dist = distance(c1.center, c2.center)
        # return center_dist < (c1.radius + c2.radius)
        c1 = geom.Point(self.center.x, self.center.y).buffer(self.radius)
        c2 = geom.Point(other_circle.center.x, other_circle.center.y).buffer(other_circle.radius)
        return c1.intersects(c2)
    
    def within(self, shape: Union['Rectangle', 'Circle']) -> bool:
        """
        Check if this circle is completely within a rectangle or another circle.

        Args:
            shape (Rectangle or Circle): The shape to check against.

        Returns:
            bool: True if this circle is completely within the shape, False otherwise.
        """
        circle_geom = geom.Point(self.center.x, self.center.y).buffer(self.radius)
        
        if isinstance(shape, Rectangle):
            other_geom = geom.box(
                shape.corners[0].x,
                shape.corners[0].y,
                shape.corners[2].x,
                shape.corners[2].y
            )
        
        elif isinstance(shape, Circle):
            other_geom = geom.Point(shape.center.x, shape.center.y).buffer(shape.radius)
    
        else:
            raise TypeError("Argument must be a Rectangle object.")
        
        return circle_geom.covered_by(other_geom)
        
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
        """
        Accepts:
          Rectangle(width, height, [center], [rotation])
          Rectangle(width=..., height=..., center=..., rotation=...)
          Rectangle(width, height, center=(x,y), rotation=...)
        """
        # 1) Merge positionals into kwargs (without clobbering provided keywords)
        if len(args) >= 1 and 'width' not in kwargs:
            kwargs['width'] = args[0]
        if len(args) >= 2 and 'height' not in kwargs:
            kwargs['height'] = args[1]
        if len(args) >= 3 and 'center' not in kwargs:
            kwargs['center'] = args[2]
        if len(args) >= 4 and 'rotation' not in kwargs:
            kwargs['rotation'] = args[3]

        # 2) Defaults
        kwargs.setdefault('width', None)
        kwargs.setdefault('height', None)
        kwargs.setdefault('center', Point(0, 0))
        kwargs.setdefault('rotation', 0.0)
        
        # 3) Validate required fields
        if kwargs['width'] is None or kwargs['height'] is None:
            raise TypeError("Rectangle requires width and height.")

        # 4) Normalize center
        center = kwargs['center']
        if isinstance(center, tuple):
            center = Point(*center)
        kwargs['center'] = center

        # 5) Build params and assign
        params = RectangleParams(**kwargs)
        self.width = params.width
        self.height = params.height
        self._center = params.center
        self._rotation = params.rotation
        
    @property
    def center(self) -> Point:
        return self._center
    
    @center.setter
    def center(self, new_center: Union[Point, Tuple]) -> Point:
        if isinstance(new_center, tuple):
            self._center = Point(*new_center)
        elif isinstance(new_center, Point):
            self._center = new_center
        else:
            raise TypeError("center must be a tuple (x, y) or a Point object.")
        return self._center
       
    def translate(self, translation: Tuple[float, float]) -> 'Rectangle':
        """Translate the rectangle by the given vector (dx, dy).
        This method moves the rectangle to a new position by adding the translation vector to its current center.
        
        Args:
            translation (Tuple[float, float]): The translation vector (dx, dy).
            
        Returns:
            Rectangle: The translated rectangle.
        """
        if isinstance(translation, tuple):
            dx, dy = translation
        else:
            raise TypeError("translation must be a tuple (dx, dy).")
        
        self._center = Point(self._center.x + dx, self._center.y + dy)
        return self
        
    @property
    def rotation(self) -> float:
        return self._rotation
    
    def rotate(self, angle_deg: float, pivot: Optional[Union[Point, Tuple]] = None) -> 'Rectangle':
        trans_params = TransParams(angle_deg=angle_deg)
        if isinstance(pivot, tuple):
            pivot = Point(*pivot)
        trans_params.pivot = pivot if pivot else self.center 
        self._center = rotate2D(self._center, trans_params)
        self._rotation += angle_deg
        
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
    
    def section_modulus(self):
        """Calculate the elastic section modulus about the strong axis (x-axis)"""
        ix, _, _ = self.moment_inertia(angle_deg=0)
        sx_top = ix / (self.height / 2)
        sx_bot = ix / (self.height / 2)
        return {"top": sx_top, "bottom": sx_bot}
    
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
    
    @property
    def x_min(self):
        """Returns the minimum x coordinate of the section."""
        return min([corner.x for corner in self.corners])
    
    @property
    def x_max(self):
        """Returns the maximum x coordinate of the section."""
        return max([corner.x for corner in self.corners])
    
    @property
    def y_min(self):
        """Returns the minimum y coordinate of the section."""
        return min([corner.y for corner in self.corners])
    
    @property
    def y_max(self):
        """Returns the maximum y coordinate of the section."""
        return max([corner.y for corner in self.corners])  
    
    def within(self, shape: Union['Rectangle', 'Circle']) -> bool:
        """
        Check if this rectangle is completely within another rectangle or a circle.

        Args:
            shape (Rectangle or Circle): The shape to check against.

        Returns:
            bool: True if this rectangle is completely within the shape, False otherwise.
        """
        rect_geom = geom.box(
            self.corners[0].x,
            self.corners[0].y,
            self.corners[2].x,
            self.corners[2].y
        )

        if isinstance(shape, Rectangle):
            other_geom = geom.box(
                shape.corners[0].x,
                shape.corners[0].y,
                shape.corners[2].x,
                shape.corners[2].y
            )
            
        elif isinstance(shape, Circle):
            other_geom = Point(shape.center.x, shape.center.y).buffer(shape.radius)

        else:
            raise TypeError("Argument must be a Rectangle or Circle object.")

        return rect_geom.covered_by(other_geom)
    
    def get_net_area(self, circles: List[Circle]) -> float:
        """Compute the net area of the rectangle after subtracting the area of the overlapping circles.

        Args:
            circles List(Circle): List of Circle objects to be subtracted from the rectangle.

        Returns:
            float: Net area of the rectangles after subtracting the area of the overlapping circle.
        """
        
        if not all(isinstance(circle, Circle) for circle in circles):
            raise TypeError("circles must be a list of Circle objects.")
        
        rect_geom = geom.box(
            self.corners[0].x, 
            self.corners[0].y, 
            self.corners[2].x, 
            self.corners[2].y
        )
        net_area = rect_geom.area
        
        if not circles: # Empty list of circles
            return net_area
        else:
            for circle in circles:
                circle_geom = geom.Point(circle.center.x, circle.center.y).buffer(circle.radius)
                net_area -= rect_geom.intersection(circle_geom).area
            return net_area
    
    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height}, center={self.center}, rotation={self.rotation})"
    
    def __str__(self):
        return f"Rectangle(width={self.width:.3f}, height={self.height:.3f}, center={self.center}, rotation={self.rotation:.3f})"


@dataclass
class TSectionParams:
    total_height: float
    web_width: float
    flange_width: float
    flange_thickness: float
    center: "Point" = field(default_factory=lambda: Point(0, 0))
    rotation: float = 0.0  
    

class TSection:
    def __init__(self, *args, **kwargs) -> None:
        """
        TSection(total_height, web_width, flange_width, flange_thickness, [center], [rotation])
        or keyword args.
        """

        # 1) Merge positionals into kwargs without overwriting provided keywords
        if len(args) >= 1 and "total_height" not in kwargs:
            kwargs["total_height"] = args[0]
        if len(args) >= 2 and "web_width" not in kwargs:
            kwargs["web_width"] = args[1]
        if len(args) >= 3 and "flange_width" not in kwargs:
            kwargs["flange_width"] = args[2]
        if len(args) >= 4 and "flange_thickness" not in kwargs:
            kwargs["flange_thickness"] = args[3]
        if len(args) >= 5 and "center" not in kwargs:
            kwargs["center"] = args[4]
        if len(args) >= 6 and "rotation" not in kwargs:
            kwargs["rotation"] = args[5]
            
        # 2) Defaults
        kwargs.setdefault("center", Point(0, 0))
        kwargs.setdefault("rotation", 0.0)

        # 3) Normalize center
        center = kwargs["center"]
        if isinstance(center, tuple):
            center = Point(*center)
        kwargs["center"] = center
        
        # 4) Build params dataclass (raises if required fields missing)
        params = TSectionParams(**kwargs)

        # 5) Build + align (components should be created using params.rotation)
        self._create_components(params)
        self._align_centroid_to_origin()
        self._rotation = params.rotation

        # 6) Translate to desired center
        self.center = params.center
        
    def _create_components(self, params: TSectionParams) -> None:
        if params.flange_width < params.web_width:
            raise ValueError("flange_width must be ≥ web_width for a T-section.")

        # Web: full height, centered initially, pass rotation here
        self._web: Rectangle = Rectangle(
            params.web_width,
            params.total_height,
            center=Point(0, 0),
            rotation=params.rotation
        )

        # Overhang width on each side
        overhang_total = params.flange_width - params.web_width
        overhang_each = overhang_total / 2.0

        # y-center of flange rectangles (top strip)
        y_flange = 0.5 * (params.total_height - params.flange_thickness)

        self._flange_left = None
        self._flange_right = None

        if overhang_each > 0:
            # x-centers of left/right overhangs
            x_left  = - (params.web_width / 2.0 + overhang_each / 2.0)
            x_right = + (params.web_width / 2.0 + overhang_each / 2.0)

            # Pass rotation here too
            self._flange_left: Rectangle = Rectangle(
                overhang_each,
                params.flange_thickness,
                center=Point(x_left, y_flange),
                rotation=params.rotation
            )
            self._flange_right: Rectangle = Rectangle(
                overhang_each,
                params.flange_thickness,
                center=Point(x_right, y_flange),
                rotation=params.rotation
            )
        # If overhang_each == 0 → no separate flange rectangles (pure rectangular web)

        
    def _align_centroid_to_origin(self):
        centroid = self._get_centroid()
        for shape in self.components.values():
            shape.center = shape.center - centroid

    def _get_centroid(self):
        ax, ay = 0.0, 0.0
        for shape in self.components.values():
            ax += shape.area * (shape.center.x - self._web.center.x)
            ay += shape.area * (shape.center.y - self._web.center.y)
        return Point(ax / self.area, ay / self.area)
    
    @property
    def yt(self) -> float:
        """Distance from centroid of section to top fiber."""
        return self._web.height / 2 - self._get_centroid().y
    
    @property
    def yb(self) -> float:
        """Distance from centroid of section to bottom fiber."""
        return self._web.height / 2 + self._get_centroid().y
    
    @property
    def components(self) -> Dict[str, Rectangle]:
        comps = {"web": self._web}
        if self._flange_left is not None:
            comps["flange_left"] = self._flange_left
        if self._flange_right is not None:
            comps["flange_right"] = self._flange_right
        return comps
               
    @property
    def area(self) -> float:
        return sum(shape.area for shape in self.components.values())

    @property
    def center(self) -> "Point":
        centroid = self._get_centroid()
        return self._web.center + centroid

    @center.setter
    def center(self, new_center: Union["Point", Tuple[float, float]]) -> "Point":
        if isinstance(new_center, tuple):
            new_center = Point(*new_center)
            
        current_centroid = self.center
        dx = new_center.x - current_centroid.x
        dy = new_center.y - current_centroid.y
        
        for shape in self.components.values():
            shape.translate((dx, dy))
            
        return new_center

    def moment_inertia(self, angle_deg: float = 0, pivot: Optional[Union["Point", Tuple[float, float]]] = None) -> Tuple[float, float, float]:
        ix_rt_total = iy_rt_total = ixy_rt_total = 0.0
        for shape in self.components.values():
            ix_rt, iy_rt, ixy_rt = shape.moment_inertia(angle_deg=angle_deg, pivot=pivot)
            area = shape.area
            ix_rt_total += ix_rt + area * (self.center.y - shape.center.y)**2
            iy_rt_total += iy_rt + area * (self.center.x - shape.center.x)**2
            ixy_rt_total += ixy_rt + area * (self.center.y - shape.center.y) * (self.center.x - shape.center.x)
        return ix_rt_total, iy_rt_total, ixy_rt_total
    
    def section_modulus(self):
        """Calculate the elastic section modulus about the strong axis (x-axis)"""
        ix, _, _ = self.moment_inertia(angle_deg=0)
        sx_top = ix / self.yt
        sx_bot = ix / self.yb
        return {"top": sx_top, "bottom": sx_bot}
        

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
    def Ix(self) -> float:
        """Return the moment of inertia about the x-axis [in4]"""
        return self.props()['Ix']
    
    @property
    def Iy(self) -> float:
        """Return the moment of inertia about the y-axis [in4]"""
        return self.props()['Iy']
    
    @property
    def Zx(self) -> float:
        """Return the plastic section modulus about the x-axis [in3]"""
        return self.props()['Zx']
    
    @property
    def Zy(self) -> float:
        """Return the plastic section modulus about the y-axis [in3]"""
        return self.props()['Zy']
    
    @property
    def rx(self) -> float:
        """Return the radius of gyration about the x-axis [in]"""
        return self.props()['rx']

    @property
    def ry(self) -> float:
        """Return the radius of gyration about the y-axis [in]"""
        return self.props()['ry']

    @property
    def Awy(self) -> float:
        """Return the web area of the W-section [in2]"""
        return self.d * self.tw
    
    @property
    def T(self) -> float:
        """Returns the height of the web between flange fillets [in]"""
        return self.props()['T']
    
    @property
    def kdes(self) -> float:
        """Return the distance from outer face of flange to web toe of fillet [in]"""
        return self.props()['kdes']
    
    @property
    def kdet(self) -> float:
        """Return the distance from outer face of flange to web toe of fillet [in]"""
        return self.props()['kdet']
    
    @property
    def k1(self) -> float:
        """Return the distance from middle of flange to flange toe of fillet [in]"""
        return self.props()['k1']
    
    @property
    def htw(self) -> float:
        """Return the ratio of web slenderness ratio (h/tw)"""
        return self.props()['h/tw']
    
    @property
    def bf2tf(self) -> float:
        """Return the ratio of flange slenderness ratio (bf/2tf)"""
        return self.props()['bf/2tf']
    
    
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
    """
    Loads the AISC shapes database Excel file, automatically searching upward
    from the current script location until it finds the file.
    """
    filename = 'aisc-shapes-database-v16.0.xlsx'
    relative_path = Path("assets/data/structure") / filename
    
    # start searching upward from where this script lives
    current_dir = Path(__file__).resolve().parent
    
    for parent in [current_dir, *current_dir.parents]:
        potential_path = parent / relative_path
        if potential_path.is_file():
            file_name = potential_path.name
            break
    else:
        raise FileNotFoundError(f"Could not find {filename} in any parent directories.")
    
    return pd.read_excel(potential_path, sheet_name='Database v16.0')    


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


def get_circle_group_centroid(circles: List[Circle]) -> Point:
    """Returns the centroid of a group of cicles."""
    if not circles:
        return None
    
    total_area = sum([circle.area for circle in circles])
    if total_area == 0:
        return Point(0, 0)
    
    x_sum = sum([circle.center.x * circle.area for circle in circles])
    y_sum = sum([circle.center.y * circle.area for circle in circles])
    
    return Point(x_sum / total_area, y_sum / total_area)


def parabola_from_points(p0: Point, p1: Point, y_ext: float, **kwargs) -> Tuple[float, float, float]:
    """Returns the coefficients of a parabola defined by three points, two end points and the orinate of extreme point (y-coordinate). No need to provide the x coordinate of extreme point. The parabola is in the form of y = ax^2 + bx + c.

    Args:
        p0 (Point): coordinates of first end point.
        p1 (Point): coordinates of second end point.
        y_ext (float): ordinate of extreme point.

    Returns:
        Tuple[float, float, float]: coefficients a, b, c of the parabola equation y = ax^2 + bx + c.
    """
    
    # Define the system of equations
    def equations(vars: Tuple[float, float, float]) -> Tuple[float, float, float]:
        a, b, c = vars
        eq1 = a * p0.x**2 + b * p0.x + c - p0.y
        eq2 = a * p1.x**2 + b * p1.x + c - p1.y
        e_ext = -b / (2 * a)
        eq3 = a * e_ext**2 + b * e_ext + c - y_ext
        return (eq1, eq2, eq3)
    
    # Solve the system of equations
    initial_guess = kwargs.get('initial_guess', (1, 1, 1))
    a, b, c = fsolve(equations, initial_guess)
    
    return a, b, c


# def get_rectangle_net_area(rectangles: List[Rectangle], circles: List[Circle], show_plot: bool = True) -> float:
#     """Compute the net area of rectangles after subtracting the area of the overlapping circle.

#     Args:
#         rectangles List(Rectangle): List of Rectangle objects.
#         circles List(Circle): List of Circle objects to be subtracted from the rectangle.

#     Returns:
#         float: Net area of the rectangles after subtracting the area of the overlapping circle.
#     """
#     if not all(isinstance(rect, Rectangle) for rect in rectangles):
#         raise TypeError("rectangles must be a list of Rectangle objects.")
    
#     if not all(isinstance(circle, Circle) for circle in circles):
#         raise TypeError("circles must be a list of Circle objects.")
    
#     if circles is None:
#         circles = []
#     circle_geoms = [
#         geom.Point(circle.center.x, circle.center.y).buffer(circle.radius)
#         for circle in circles
#     ]
    
#     net_areas = []
    
#     fig, ax = plt.subplots() if show_plot else (None, None)

#     for rect in rectangles:
#         r = geom.box(
#             rect.corners[0].x, 
#             rect.corners[0].y, 
#             rect.corners[2].x, 
#             rect.corners[2].y
#         )
#         net_area = r.area
        
#         for c in circle_geoms:
#             net_area -= r.intersection(c).area
            
#         net_areas.append(net_area)

#         if show_plot:
#             # Plot rectangle
#             x, y = r.exterior.xy
#             ax.fill(x, y, edgecolor='black', fill=False)
            
#     if show_plot:
#         for c in circle_geoms:
#             x, y = c.exterior.xy
#             ax.fill(x, y, color='red', alpha=0.3)

#         ax.set_aspect('equal')
#         ax.set_title("Rectangles and Rebars (Circles)")
#         plt.show()
        
#     return np.array([net_areas])
    
    
# Main function =========================================================
def main():
    pass
    
    
if __name__ == "__main__":
    main()