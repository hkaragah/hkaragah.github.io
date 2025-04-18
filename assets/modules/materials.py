
from typing import Protocol, Union
import numpy as np
from scipy.optimize import curve_fit, root_scalar


class Material(Protocol):
    name:str
    eps_u: float
    
    def stress(self, eps: float) -> float:
        pass


class Steel:
    def __init__(self, name: str, fy: float, fu: float) -> None:
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
    """
    Bilinear steel material with strain hardening.
    """

    def __init__(self, name: str, fy: float, fu: float, alpha: float=0.01) -> None:
        super().__init__(name, fy, fu)
        self.alpha = alpha  # strain hardening ratio
    
    @property
    def eps_y(self) -> float:
        return self.fy / self.E  # Yield strain

    @property
    def eps_u(self) -> float:
        if self.alpha == 0:
            return 0.05
        else:
            return self.eps_y + (self.fu - self.fy) / (self.E * self.alpha)  # Ultimate strain

    def stress(self, eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the stress based on the strain using the bilinear model.
        """
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
        
        
        
def main():
    pass
    
    
if __name__ == "__main__":
    main()