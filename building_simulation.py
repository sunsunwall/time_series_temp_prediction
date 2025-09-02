# Building Indoor Climate Simulation
# This script simulates indoor temperature and humidity for an office building

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BuildingParameters:
    """Building thermal parameters for a typical office building"""
    def __init__(self):
        # Building geometry
        self.floor_area = 1000  # m²
        self.ceiling_height = 3.0  # m
        self.volume = self.floor_area * self.ceiling_height  # m³
        
        # Thermal properties
        self.U_wall = 0.3  # W/m²K (U-value for walls)
        self.U_roof = 0.25  # W/m²K (U-value for roof)
        self.U_window = 1.2  # W/m²K (U-value for windows)
        
        # Surface areas (assuming square building)
        self.wall_area = 4 * np.sqrt(self.floor_area) * self.ceiling_height  # m²
        self.roof_area = self.floor_area  # m²
        self.window_area = 0.2 * self.wall_area  # 20% of wall area
        
        # Thermal mass
        self.thermal_mass = 200000  # J/K (building thermal mass)
        
        # HVAC system
        self.hvac_capacity = 50000  # W (heating/cooling capacity)
        self.hvac_efficiency = 0.8  # efficiency factor
        
        # Internal heat gains
        self.occupancy_heat = 100  # W/person
        self.equipment_heat = 15  # W/m²
        self.lighting_heat = 10  # W/m²
        self.occupancy_density = 0.1  # persons/m²
        
        # Air properties
        self.air_density = 1.225  # kg/m³
        self.air_heat_capacity = 1006  # J/kgK
        self.ventilation_rate = 0.5  # air changes per hour

class BuildingThermalModel:
    """Building thermal and humidity simulation model"""
    
    def __init__(self, building_params):
        self.building = building_params
        
    def calculate_solar_gains(self, solar_irradiance, hour):
        """Calculate solar heat gains through windows"""
        # Solar heat gain coefficient for windows
        SHGC = 0.6
        
        # Solar angle factor (simplified - varies with time of day)
        # Peak at noon (hour 12), minimum at night
        solar_angle_factor = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        
        solar_gains = solar_irradiance * self.building.window_area * SHGC * solar_angle_factor
        return solar_gains
    
    def calculate_internal_gains(self, hour, is_weekday=True):
        """Calculate internal heat gains from occupancy, equipment, and lighting"""
        # Occupancy pattern (higher during office hours)
        if is_weekday and 8 <= hour <= 18:
            occupancy_factor = 1.0
        elif is_weekday and (7 <= hour < 8 or 18 < hour <= 19):
            occupancy_factor = 0.5
        else:
            occupancy_factor = 0.1
            
        # Equipment and lighting (on during occupied hours)
        equipment_factor = 1.0 if occupancy_factor > 0.5 else 0.3
        lighting_factor = 1.0 if occupancy_factor > 0.5 else 0.1
        
        occupancy_gains = (self.building.occupancy_heat * 
                          self.building.occupancy_density * 
                          self.building.floor_area * occupancy_factor)
        
        equipment_gains = (self.building.equipment_heat * 
                          self.building.floor_area * equipment_factor)
        
        lighting_gains = (self.building.lighting_heat * 
                         self.building.floor_area * lighting_factor)
        
        return occupancy_gains + equipment_gains + lighting_gains
    
    def calculate_hvac_load(self, indoor_temp, setpoint_temp, outdoor_temp, hvac_mode):
        """Calculate HVAC heating/cooling load"""
        temp_diff = setpoint_temp - indoor_temp
        
        if hvac_mode == 'Heating' and temp_diff > 0.5:
            # Heating needed
            hvac_load = min(self.building.hvac_capacity, 
                           temp_diff * self.building.thermal_mass * 0.1)
        elif hvac_mode == 'Cooling' and temp_diff < -0.5:
            # Cooling needed
            hvac_load = -min(self.building.hvac_capacity, 
                            abs(temp_diff) * self.building.thermal_mass * 0.1)
        else:
            hvac_load = 0
            
        return hvac_load * self.building.hvac_efficiency
    
    def calculate_transmission_loss(self, indoor_temp, outdoor_temp):
        """Calculate heat loss through building envelope"""
        temp_diff = indoor_temp - outdoor_temp
        
        # Total U-value weighted by area
        total_UA = (self.building.U_wall * (self.building.wall_area - self.building.window_area) +
                   self.building.U_roof * self.building.roof_area +
                   self.building.U_window * self.building.window_area)
        
        transmission_loss = total_UA * temp_diff
        return transmission_loss
    
    def calculate_ventilation_loss(self, indoor_temp, outdoor_temp, outdoor_humidity):
        """Calculate heat loss through ventilation"""
        temp_diff = indoor_temp - outdoor_temp
        
        # Ventilation heat loss
        ventilation_flow = (self.building.ventilation_rate * 
                           self.building.volume / 3600)  # m³/s
        
        ventilation_loss = (ventilation_flow * 
                           self.building.air_density * 
                           self.building.air_heat_capacity * 
                           temp_diff)
        
        return ventilation_loss
    
    def calculate_humidity_balance(self, indoor_humidity, outdoor_humidity, indoor_temp, outdoor_temp, hour):
        """Calculate indoor humidity balance"""
        # Moisture generation from occupancy (simplified)
        if 8 <= hour <= 18:  # Office hours
            moisture_generation = 0.05  # kg/h per person
        else:
            moisture_generation = 0.01  # kg/h per person
            
        total_moisture_generation = (moisture_generation * 
                                    self.building.occupancy_density * 
                                    self.building.floor_area)
        
        # Ventilation moisture exchange
        ventilation_flow = (self.building.ventilation_rate * 
                           self.building.volume / 3600)  # m³/s
        
        # Moisture content difference (kg/m³)
        # Simplified: assume constant air density and use relative humidity
        moisture_exchange = (ventilation_flow * 
                            (outdoor_humidity - indoor_humidity) * 0.01)  # Simplified conversion
        
        # Humidity balance equation
        dH_dt = (total_moisture_generation + moisture_exchange) / self.building.volume
        
        return dH_dt
    
    def thermal_balance(self, t, T_indoor, outdoor_temp, solar_irradiance, 
                       setpoint_temp, hvac_mode, hour, outdoor_humidity):
        """Thermal balance equation for indoor temperature"""
        
        # Calculate all heat flows
        solar_gains = self.calculate_solar_gains(solar_irradiance, hour)
        internal_gains = self.calculate_internal_gains(hour)
        hvac_load = self.calculate_hvac_load(T_indoor, setpoint_temp, outdoor_temp, hvac_mode)
        transmission_loss = self.calculate_transmission_loss(T_indoor, outdoor_temp)
        ventilation_loss = self.calculate_ventilation_loss(T_indoor, outdoor_temp, outdoor_humidity)
        
        # Thermal balance equation
        dT_dt = (solar_gains + internal_gains + hvac_load - 
                transmission_loss - ventilation_loss) / self.building.thermal_mass
        
        return dT_dt

def simulate_building_climate(df, thermal_model, initial_temp=20, initial_humidity=50):
    """
    Simulate indoor climate for the entire dataset
    
    Parameters:
    df: DataFrame with outdoor weather data
    thermal_model: BuildingThermalModel instance
    initial_temp: Initial indoor temperature (°C)
    initial_humidity: Initial indoor humidity (%)
    
    Returns:
    DataFrame with simulated indoor conditions
    """
    
    # Initialize arrays for indoor conditions
    indoor_temps = np.zeros(len(df))
    indoor_humidities = np.zeros(len(df))
    
    # Set initial conditions
    indoor_temps[0] = initial_temp
    indoor_humidities[0] = initial_humidity
    
    # Simulate hour by hour
    for i in range(1, len(df)):
        # Get current outdoor conditions
        outdoor_temp = df.iloc[i]['Air Temperature']
        outdoor_humidity = df.iloc[i]['Relative Humidity']
        solar_irradiance = df.iloc[i]['Global Irradiance (Swedish stations) W/m²']
        hour = pd.to_datetime(df.iloc[i]['Datetime']).hour
        
        # Get HVAC setpoint (simplified - you can enhance this)
        if 7 <= hour < 19:
            setpoint_temp = 21  # Comfort setpoint during office hours
        else:
            setpoint_temp = 17  # Night setback
            
        # Determine HVAC mode based on outdoor temperature
        if outdoor_temp <= 15:
            hvac_mode = 'Heating'
        elif outdoor_temp >= 21:
            hvac_mode = 'Cooling'
        else:
            hvac_mode = 'Off'
        
        # Calculate temperature change
        dT_dt = thermal_model.thermal_balance(
            t=i, T_indoor=indoor_temps[i-1], 
            outdoor_temp=outdoor_temp, 
            solar_irradiance=solar_irradiance,
            setpoint_temp=setpoint_temp, 
            hvac_mode=hvac_mode, 
            hour=hour, 
            outdoor_humidity=outdoor_humidity
        )
        
        # Calculate humidity change
        dH_dt = thermal_model.calculate_humidity_balance(
            indoor_humidity=indoor_humidities[i-1],
            outdoor_humidity=outdoor_humidity,
            indoor_temp=indoor_temps[i-1],
            outdoor_temp=outdoor_temp,
            hour=hour
        )
        
        # Update indoor conditions (Euler integration)
        indoor_temps[i] = indoor_temps[i-1] + dT_dt * 3600  # Convert to hourly
        indoor_humidities[i] = indoor_humidities[i-1] + dH_dt * 3600  # Convert to hourly
        
        # Apply reasonable bounds
        indoor_temps[i] = np.clip(indoor_temps[i], 15, 30)
        indoor_humidities[i] = np.clip(indoor_humidities[i], 20, 80)
    
    # Create result DataFrame
    result_df = df.copy()
    result_df['Indoor_Temperature'] = indoor_temps
    result_df['Indoor_Humidity'] = indoor_humidities
    
    return result_df

if __name__ == "__main__":
    # Load the preprocessed data
    print("Loading data...")
    df = pd.read_csv('final_df_1.0', index_col=0)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    
    # Initialize building and thermal model
    print("\nInitializing building model...")
    building = BuildingParameters()
    thermal_model = BuildingThermalModel(building)
    
    print("Building Parameters:")
    print(f"Floor area: {building.floor_area} m²")
    print(f"Volume: {building.volume} m³")
    print(f"Thermal mass: {building.thermal_mass} J/K")
    print(f"HVAC capacity: {building.hvac_capacity} W")
    
    # Simulate indoor climate
    print("\nSimulating indoor climate...")
    result_df = simulate_building_climate(df, thermal_model)
    
    # Save results
    result_df.to_csv('simulated_indoor_climate.csv', index=False)
    print(f"\nSimulation complete! Results saved to 'simulated_indoor_climate.csv'")
    
    # Show sample results
    print("\nSample results:")
    print(result_df[['Datetime', 'Air Temperature', 'Relative Humidity', 
                     'Indoor_Temperature', 'Indoor_Humidity']].head(10))
    
    # Basic statistics
    print("\nIndoor conditions statistics:")
    print(f"Indoor temperature range: {result_df['Indoor_Temperature'].min():.1f}°C to {result_df['Indoor_Temperature'].max():.1f}°C")
    print(f"Indoor humidity range: {result_df['Indoor_Humidity'].min():.1f}% to {result_df['Indoor_Humidity'].max():.1f}%")
