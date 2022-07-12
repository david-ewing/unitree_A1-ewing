"""
CODE BASED ON EXAMPLE FROM:
@misc{coumans2017pybullet,
  title={Pybullet, a python module for physics simulation in robotics, games and machine learning},
  author={Coumans, Erwin and Bai, Yunfei},
  url={www.pybullet.org},
  year={2017},
}

Example: minitaur_env_randomizer.py
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/env_randomizers/minitaur_env_randomizer.py
"""
import numpy as np
from a1_kinematics.base import env_randomizer_base

# Relative range.
a1_BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
a1_LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
# Absolute range.
BATTERY_VOLTAGE_RANGE = (7.0, 8.4)  # Unit: Volt
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.01)  # Unit: N*m*s/rad (torque/angular vel)
a1_LEG_FRICTION = (0.8, 1.5)  # Unit: dimensionless


class A1EnvRandomizer(env_randomizer_base.EnvRandomizerBase):
    """A randomizer that change the a1_gym_env during every reset."""
    def __init__(self,
                 a1_base_mass_err_range=a1_BASE_MASS_ERROR_RANGE,
                 a1_leg_mass_err_range=a1_LEG_MASS_ERROR_RANGE,
                 battery_voltage_range=BATTERY_VOLTAGE_RANGE,
                 motor_viscous_damping_range=MOTOR_VISCOUS_DAMPING_RANGE):
        self._a1_base_mass_err_range = a1_base_mass_err_range
        self._a1_leg_mass_err_range = a1_leg_mass_err_range
        self._battery_voltage_range = battery_voltage_range
        self._motor_viscous_damping_range = motor_viscous_damping_range

        np.random.seed(0)

    def randomize_env(self, env):
        self._randomize_a1(env.a1)

    def _randomize_a1(self, a1):
        """Randomize various physical properties of a1.

    It randomizes the mass/inertia of the base, mass/inertia of the legs,
    friction coefficient of the feet, the battery voltage and the motor damping
    at each reset() of the environment.

    Args:
      a1: the a1 instance in a1_gym_env environment.
    """
        base_mass = a1.GetBaseMassFromURDF()
        # print("BM: ", base_mass)
        randomized_base_mass = np.random.uniform(
            np.array([base_mass]) * (1.0 + self._a1_base_mass_err_range[0]),
            np.array([base_mass]) * (1.0 + self._a1_base_mass_err_range[1]))
        a1.SetBaseMass(randomized_base_mass[0])

        leg_masses = a1.GetLegMassesFromURDF()
        leg_masses_lower_bound = np.array(leg_masses) * (
            1.0 + self._a1_leg_mass_err_range[0])
        leg_masses_upper_bound = np.array(leg_masses) * (
            1.0 + self._a1_leg_mass_err_range[1])
        randomized_leg_masses = [
            np.random.uniform(leg_masses_lower_bound[i],
                              leg_masses_upper_bound[i])
            for i in range(len(leg_masses))
        ]
        a1.SetLegMasses(randomized_leg_masses)

        randomized_battery_voltage = np.random.uniform(
            BATTERY_VOLTAGE_RANGE[0], BATTERY_VOLTAGE_RANGE[1])
        a1.SetBatteryVoltage(randomized_battery_voltage)

        randomized_motor_damping = np.random.uniform(
            MOTOR_VISCOUS_DAMPING_RANGE[0], MOTOR_VISCOUS_DAMPING_RANGE[1])
        a1.SetMotorViscousDamping(randomized_motor_damping)

        randomized_foot_friction = np.random.uniform(a1_LEG_FRICTION[0],
                                                     a1_LEG_FRICTION[1])
        a1.SetFootFriction(randomized_foot_friction)
