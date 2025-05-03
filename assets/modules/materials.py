
from typing import Protocol, Union, Optional
import numpy as np
from scipy.optimize import curve_fit, root_scalar
from traitlets import default


class Material(Protocol):
    name:str
    eps_u: float
    
    def stress(self, eps: float) -> float:
        pass

# Concrete material classes =========================================================
class Concrete:
    def __init__(self, fc: float) -> None:
        """

        Args:
            fc (float): psi, Compressive strength
        """
        self.fc = fc
    
    @property
    def E(self) -> float:
        """

        Returns:
            float: psi, Modulus of elasticity (Ec) per ACI 318-25: 19.2.2
        """
        return 57000 * np.sqrt(self.fc)  # psi, Modulus of elasticity
    
    @property
    def fr(self) -> float:
        """

        Returns:
            float: psi, Modulus of rupture (fr) per ACI 318-25: 19.2.3
        """
        return 7.5 * np.sqrt(self.fc)  # psi, Fracture strength
    
    def stress(self, eps:Union[float, np.ndarray]) -> float:
        """Calculate the stress based on the strain using the concrete model.
        
        Args:
            eps (float): Strain value for which to calculate stress.
            
        Returns:
            float: Calculated stress value.
        """
        raise NotImplementedError("Concrete stress method not implemented.")


class ACIConcrete(Concrete):
    def __init__(self, fc: float) -> None:
        super().__init__(fc)
        self.name = "ACI Concrete"
    
    @property
    def eps_cu(self) -> float:
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
        eps_array = np.asarray(eps)
        eps_ratio = eps_array / self.eps_cu
        stress = np.where(
            eps_ratio > 1,
            0.0,
            np.where(
                eps_ratio < 1 - self.beta_1,
                0.0,
                - 0.85 * self.fc
            )
        )
        return stress.item() if np.isscalar(eps) else stress

        
    
# Steel material classes =========================================================
class Steel:
    def __init__(self, name: str, fy: float, fu: Optional[float]=None) -> None:
        self.name = name
        self.fy = fy  # ksi, Yield strength
        self.fu = fu  # ksi, Ultimate strength
    
    @property
    def E(self) -> float:
        return 29000 # ksi, Modulus of elasticity
    
    @property
    def eps_y(self) -> float:
        return self.fy / self.E
    

class BilinearSteel(Steel, Material):
    def __init__(self, name: str, fy: float, fu: Optional[float]=None, alpha: Optional[float]=None) -> None:
        """Bilinear steel material with optional strain hardening.
        
        Args:
            name (str): Name of the material.
            fy (float): Yield strength in ksi.
            fu (float): Ultimate strength in ksi.
            alpha (float): Strain hardening ratio. Default is 0.01.
        """
        super().__init__(name, fy, fu)
        self.alpha = alpha  # strain hardening ratio
    
    @property
    def eps_u(self) -> float:
        if any(value is None for value in (self.alpha, self.fu)):
            return None
        try:
            return self.eps_y + (self.fu - self.fy) / (self.E * self.alpha)
        except ZeroDivisionError:
            return 0.05

    def stress(self, eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the stress based on the strain using the bilinear model.
        
        Args:
            eps (Union[float, np.ndarray]): Strain value(s) for which to calculate stress.
            
        Returns:
            Union[float, np.ndarray]: Calculated stress value(s). Returns None if alpha or fu is not provided or is None.
        """
        if any(value is None for value in (self.alpha, self.fu)):
            return None
        return np.where(
            eps > 0, 
            np.where(
                eps < self.eps_y,
                self.E * eps,
                np.where(
                    eps < self.eps_u,
                    np.where(
                        np.isclose(self.alpha, 0), 
                        self.fy,
                        self.fy + (self.fu - self.fy) * (eps - self.eps_y) / (self.eps_u - self.eps_y)
                    ),
                    0
                )
            ), 
            np.where(
                eps > -self.eps_y,
                self.E * eps,
                np.where(
                    eps > -self.eps_u,
                    np.where(
                        np.isclose(self.alpha, 0),
                        -self.fy,
                        -self.fy + (self.fu - self.fy) * (eps + self.eps_y) / (self.eps_u + self.eps_y)
                    ),
                    0
                )
            )
        )
        
          
class BilinearA36Steel(BilinearSteel):
    """
    ASTM A36 steel material with bilinear stress-strain curve.
    """

    def __init__(self, name= "A36 Steel", fy: float=36, fu: float=58, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu, alpha)
        
        
class BilinearA992Steel(BilinearSteel):
    """
    ASTM A992 steel material with bilinear stress-strain curve.
    """

    def __init__(self, name= "A992 Steel", fy: float=50, fu: float=65, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu, alpha)
        
    
class RambergOsgoodSteel(Steel, Material):
    """ 
    Uses Ramberg-Osgood equation for stress-strain relationship.
    """
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
            

class RambergOsgoodA36Steel(RambergOsgoodSteel):
    """
    ASTM A36 steel material with Ramberg-Osgood stress-strain curve.
    """

    def __init__(self, name= "A36 Steel", fy: float=36, fu: float=58, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu, alpha)
        
        
class RambergOsgoodA992Steel(RambergOsgoodSteel):
    """
    ASTM A992 steel material with Ramberg-Osgood stress-strain curve.
    """

    def __init__(self, name= "A992 Steel", fy: float=50, fu: float=65, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu, alpha)
        
        
# Steel Rebars ==========





      
def main():
    pass
    
    
if __name__ == "__main__":
    main()