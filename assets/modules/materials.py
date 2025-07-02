#######################################
# Materials module for structural engineering
# Written by: Hossein Karagah
# Date: 2025-02-15
# Description: This module provides classes and functions for defining and working with various construction materials, including concrete and steel.
#######################################




from typing import Protocol, Union, Optional
import numpy as np
from scipy.optimize import curve_fit, root_scalar


class Material(Protocol):
    name:str
    eps_u: float
    
    def stress(self, eps: float) -> float:
        pass

# Concrete material classes =========================================================
class Concrete:
    def __init__(self, fc: float, is_lightweight: bool = False, density: float = None) -> None:
        """

        Args:
            fc (float): psi, Compressive strength
            is_lightweight (bool, optional): Whether the concrete is lightweight. Defaults to False.
            density (float, optional): pcf, Unit weight of lightweight concrete (required if is_lightweight is True).
        """
        self.fc = fc
        self.is_lightweight = is_lightweight
        self.density = density

        if self.is_lightweight and self.density is None:
            raise ValueError("Density must be provided for lightweight concrete.")

    
    @property
    def E(self) -> float:
        """Returns the modulus of elasticity (E) per ACI 318-25: 19.2.2.

        Returns:
            float: psi, Modulus of elasticity (Ec) of concrete.
        """
        if self.is_lightweight:
            return 57000 * np.sqrt(self.fc)
        else:
            return self.density ** 1.5 * 33 * np.sqrt(self.fc)
    
    @property
    def lambda_(self) -> float:
        """Returns the modification factor to reflect the reduced mechanical properties of lightweight concrete relative to normal weight concrete of the same compressive strength per ACI 318-25, section 19.2.4.

        Returns:
            float: Modification factor (lambda) for lightweight concrete. It varies between 0.75 and 1.0 based on density.
        """
        if not self.is_lightweight or self.density > 135:
            return 1.0
        elif self.is_lightweight:
            if not self.density or self.density <= 100:
                return 0.75
            else: # density between 100 and 135
                return 0.0075 * self.density
    
    @property
    def fr(self) -> float:
        """

        Returns:
            float: psi, Modulus of rupture (fr) per ACI 318-25: 19.2.3
        """
        return 7.5 * self.lambda_ * np.sqrt(self.fc)  # psi, Fracture strength
    
    def stress(self, eps:Union[float, np.ndarray]) -> float:
        """Calculate the stress based on the strain using the concrete model.
        
        Args:
            eps (float): Strain value for which to calculate stress.
            
        Returns:
            float: Calculated stress value.
        """
        raise NotImplementedError("Concrete stress method not implemented.")
    
    def __repr__(self):
        return f"Concrete Material: {self.name}, fc: {self.fc} psi, E: {self.E} psi, fr: {self.fr} psi"
    
    def __str__(self):
        return f"Concrete Material: {self.name}, fc: {self.fc} psi, E: {self.E} psi, fr: {self.fr} psi"


class ACIConcrete(Concrete):
    def __init__(self, fc: float, is_lightweight: bool = False, density: float = None) -> None:
        super().__init__(fc)
        self.name = "ACI Concrete"
    
    @property
    def eps_u(self) -> float:
        """Ultimate strain (eps_u) for concrete.
        This is a constant value of 0.003 as per ACI 318-25:22.2.2.1.

        Returns:
            float: Ultimate strain (eps_cu) for concrete.
        """
        return -0.003
    
    @property
    def beta_1(self) -> float:
        """Computes the beta_1 factor for concrete based on the compressive strength (fc).
        The beta_1 factor is used in the design of reinforced concrete structures and is defined in ACI 318-25:22.2.2.4.3.

        Raises:
            ValueError: If the compressive strength (fc) is not within the specified range.

        Returns:
            float: beta_1 factor for concrete.
        """
        if 2500 <= self.fc <= 4000:
            return 0.85
        elif 4000 < self.fc <= 8000:
            return 0.85 - 0.05 * ((self.fc - 4000) / 1000)
        elif self.fc >= 8000:
            return 0.65
        else:
            raise ValueError("Concrete compressive strength (fc) must be within 2500 psi to 8000 psi.")
        
    def stress(self, eps:Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if not isinstance(eps, (float, np.ndarray)):
            raise TypeError("Strain (eps) must be a float or numpy array.")
        
        if np.isscalar(eps):
            eps = np.array([eps])
            
        eps_array = np.asarray(eps)
        eps_ratio = eps_array / self.eps_u
        stress = np.where(
            eps_ratio > 1.001,
            0.0,
            np.where(
                eps_ratio < 1 - self.beta_1,
                0.0,
                - 0.85 * self.fc
            )
        )
        
        return stress.item() if np.isscalar(eps) else stress
    
    def __repr__(self):
        return f"Concrete Material: {self.name}, fc: {self.fc} psi, E: {self.E} psi, fr: {self.fr:.2f} psi, eps_u: {self.eps_u:.2e}, beta_1: {self.beta_1}"

    def __str__(self):
        return f"Concrete Material: {self.name}, fc: {self.fc} psi, E: {self.E} psi, fr: {self.fr:.2f} psi, eps_u: {self.eps_u:.2e}, beta_1: {self.beta_1}"
        
    
# Steel material classes =========================================================
class Steel:
    def __init__(self, name: str, fy: float, fu: Optional[float]=None, alpha:Optional[float]=None) -> None:
        """

        Args:
            name (str): ASTM designation of the steel material (e.g., "A36", "A992").
            fy (float): Yield strength in psi.
            fu (Optional[float], optional): Ultimate strength in psi. Defaults to None.
            alpha (Optional[float], optional): Strain hardening ratio. Defaults to None.
        """
        self.name = name
        self.fy = fy
        self.fu = fy if fu is None else fu
        self.alpha = alpha if alpha is not None else 0.0
        self._nu = 0.25 # Poisson's ratio for steel
        self._E = 29e6 # psi, Modulus of elasticity for steel
        
    @property
    def E(self) -> float:
        "Returns modulus of elasticity for steel in psi."
        return self._E
    
    @property
    def nu(self) -> float:
        "Returns poisson's ratio for steel."
        return 0.25
    
    @nu.setter
    def nu(self, new_value: float) -> None:
        """Set the Poisson's ratio for steel.
        
        Args:
            value (float): Poisson's ratio new value.
        """
        if not (0 < new_value < 0.5):
            raise ValueError("Poisson's ratio must be between 0 and 0.5.")
        self._nu = new_value
    
    @property
    def G(self) -> float:
        "Returns shear modulus for steel."
        return self._E / (2 * (1 + self._nu))
    
    @property
    def eps_y(self) -> float:
        "Returns strain at yield point (eps_y) for steel."
        return self.fy / self.E
    
    @property
    def eps_u(self) -> float:
        """Ultimate strain (eps_u) for steel.
        This is calculated based on the yield strain (eps_y), ultimate strength (fu), and strain hardening ratio (alpha).
        The formula used is: eps_u = eps_y + (fu - fy) / (E * alpha).
        If alpha is 0, eps_u is set to 0.05 as a default value.
        """
        return 0.05 if self.alpha==0.0 else self.eps_y + (self.fu - self.fy) / (self.E * self.alpha)
        
    def __repr__(self):
        return f"Steel Material: {self.name}, fy: {self.fy} psi, fu: {self.fu} psi, E: {self.E} psi, eps_y: {self.eps_y:.2e}, eps_u: {self.eps_u:.2e}, alpha: {self.alpha}"
    
    def __str__(self):
        return f"Steel Material: {self.name}, fy: {self.fy} psi, fu: {self.fu} psi, E: {self.E} psi, eps_y: {self.eps_y:.2e}, eps_u: {self.eps_u:.2e}, alpha: {self.alpha}"
        
class BilinearSteel(Steel, Material):
    def __init__(self, name: str, fy: float, fu: Optional[float]=None, alpha: Optional[float]=None) -> None:
        """Bilinear steel material.
        
        Args:
            name (str): ASTM designation of the steel material (e.g., "A36", "A992").
            fy (float): Yield strength in psi.
            fu (Optional[float], optional): Ultimate strength in psi. Defaults to None.
            alpha (Optional[float], optional): Strain hardening ratio. Defaults to None.
        """
        super().__init__(name, fy, fu, alpha)

    def stress(self, eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the stress based on the strain using the bilinear model.
        
        Args:
            eps (Union[float, np.ndarray]): Strain value(s) for which to calculate stress.
            
        Returns:
            Union[float, np.ndarray]: Calculated stress value(s) in psi.
        """
        return np.where( # Tension
            eps > 0, 
            np.where(
                eps < self.eps_y, # Elastic region
                self.E * eps,
                np.where(
                    eps < self.eps_u, # Plastic region
                    np.where(
                        np.isclose(self.alpha, 0), 
                        self.fy,
                        self.fy + (self.fu - self.fy) * (eps - self.eps_y) / (self.eps_u - self.eps_y)
                    ),
                    0 # Post-ultimate region
                )
            ), 
            np.where( # Compression
                eps > -self.eps_y, # Elastic region
                self.E * eps,
                np.where(
                    eps > -self.eps_u, # Plastic region
                    np.where(
                        np.isclose(self.alpha, 0),
                        -self.fy,
                        -self.fy + (self.fu - self.fy) * (eps + self.eps_y) / (self.eps_u + self.eps_y)
                    ),
                    0 # Post-ultimate region
                )
            )
        )
        
          
class BilinearA36Steel(BilinearSteel):
    "ASTM A36 steel material with bilinear stress-strain curve."
    def __init__(self, name= "A36", fy: float=36e3, fu: float=58e3, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu, alpha)
        
        
class BilinearA992Steel(BilinearSteel):
    "ASTM A992 steel material with bilinear stress-strain curve."
    def __init__(self, name= "A992", fy: float=50e3, fu: float=65e3, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu, alpha)
    
class RambergOsgoodSteel(Steel, Material):
    "Uses Ramberg-Osgood equation for stress-strain relationship."
    def __init__(self, name:str, fy: float, fu: float, K: float, n:float, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu)
        self.alpha = alpha  # strain hardening ratio
        self.K = K  # nonlinear modulus
        self.n = n # strain-hardening exponent
    
    @property
    def eps_y(self) -> float:
        return self.fy / self.E  # Yield strain
    
    @property
    def eps_u(self) -> float:
        return self.eps_y + (self.fu - self.fy) / (self.E * self.alpha)
    
    def ramberg_osgood(self, stress: float, K: float, n: float) -> float:
        return stress / self.E + np.power(stress / K, n)
    
    def fit_params(self, stress: np.ndarray, strain: np.ndarray, p0) -> tuple:
        params, _ = curve_fit(self.ramberg_osgood, stress, strain, p0)
        return params # K, n
    
    def stress(self, eps: Union[float, np.ndarray]) -> float:
        """
        Calculate the stress based on the strain using the Ramberg-Osgood model.
        """
        # p0 = [self.fy, 5] # Initial guess for K and n
        # stress = np.array([0, self.fy, self.fu])
        # strain = np.array([0, self.eps_y, self.eps_u])
        # K, n = self.fit_params(stress, strain, p0)
        
        if np.isscalar(eps):
            return np.array([eps])
        
        def func(stress, strain):
            return stress / self.E + self.K * (stress / self.E)**self.n - strain

        stresses = np.zeros_like(eps)
        
        for i, strain in enumerate(eps):
            init_sigma_guess = strain * self.E
            result = root_scalar(func, args=(strain,), bracket=[0, init_sigma_guess * 2], method='brentq')
            
            if result.converged:
                stresses[i] = result.root
            else:
                raise RuntimeError(f"Failed to converge for strain = {strain}")

        return stresses
            
        
        
# Steel Rebars ==========





      
def main():
    pass
    
    
if __name__ == "__main__":
    main()