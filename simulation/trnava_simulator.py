import os
import traci
import xml.etree.ElementTree as ET


class SumoSimulationAPI:
    def __init__(self, config_file, sumo_binary="sumo-gui", scale_factor=288):
        """
        Initializes the SumoSimulationAPI.

        Args:
            config_file (str): Path to the SUMO configuration file (.sumocfg).
            sumo_binary (str): The SUMO binary to use ("sumo" for CLI, "sumo-gui" for GUI).
            scale_factor (int): The number of simulation seconds to represent each real second.
        """
        self.config_file = os.path.expanduser(config_file)
        self.sumo_binary = sumo_binary
        self.scale_factor = scale_factor
        self.step_length = 1  # Default to 1 second if not in the config file
        self.simulation_duration = None
        self.running = False
        self._parse_config_file()

    def _parse_config_file(self):
        """
        Parses the SUMO configuration file to extract step length and simulation duration.
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")

        try:
            tree = ET.parse(self.config_file)
            root = tree.getroot()

            # Extract step length
            time_element = root.find(".//time")
            if time_element is not None:
                self.step_length = float(time_element.get("step-length", self.step_length))

            # Extract simulation duration (begin and end times)
            if time_element is not None:
                begin = float(time_element.get("begin", 0))
                end = float(time_element.get("end", 0))
                self.simulation_duration = end - begin if end > begin else None

        except ET.ParseError as e:
            raise RuntimeError(f"Failed to parse the configuration file: {e}")

    def start_simulation(self):
        """
        Starts the SUMO simulation with TraCI.
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")

        try:
            traci.start([self.sumo_binary, "-c", self.config_file, "--step-length", str(self.step_length)])
            self.running = True
            print(f"SUMO simulation started with TraCI. Step length: {self.step_length}")
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO simulation: {e}")

    def stop_simulation(self):
        """
        Stops the SUMO simulation.
        """
        if self.running:
            try:
                traci.close()
                self.running = False
                print("SUMO simulation stopped successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to stop SUMO simulation: {e}")

    def simulation_step(self):
        """
        Advances the simulation by the scale factor (representing scaled seconds).
        """
        if not self.running:
            raise RuntimeError("Simulation is not running. Call start_simulation() first.")

        try:
            traci.simulationStep()
        except Exception as e:
            raise RuntimeError(f"Failed to advance simulation step: {e}")

    def get_traffic_light_ids(self):
        """
        Retrieves all traffic light IDs in the simulation.

        Returns:
            list: List of traffic light IDs.
        """
        return traci.trafficlight.getIDList()

    def set_light(self, traffic_light_id, phase):
        """
        Sets the phase of a traffic light.

        Args:
            traffic_light_id (str): The ID of the traffic light.
            phase (int): The desired phase to set.
        """
        if traffic_light_id not in traci.trafficlight.getIDList():
            raise ValueError(f"Traffic light ID '{traffic_light_id}' does not exist.")

        try:
            traci.trafficlight.setPhase(traffic_light_id, phase)
            print(f"Traffic light '{traffic_light_id}' set to phase {phase}.")
        except Exception as e:
            raise RuntimeError(f"Failed to set phase for traffic light '{traffic_light_id}': {e}")


if __name__ == "__main__":
    CONFIG_FILE = "data/sumo_network/osm.sumocfg"

    # Initialize the API
    api = SumoSimulationAPI(CONFIG_FILE)

    # Define the number of time steps
    N = 3600  # Number of simulation steps (equivalent to 1 hour if step-length = 1 second)

    try:
        # Start the simulation
        api.start_simulation()

        # Simulation loop
        for step in range(N):
            print(f"Simulation Step {step + 1} (Simulated Time: {step * api.step_length} seconds):")
            api.simulation_step()

            # Example API usage
            traffic_lights = api.get_traffic_light_ids()
            print(traffic_lights)

            if traffic_lights:
                # Set phase for the first traffic light between Cukrovar and Biely Kostol
                first_light = traffic_lights[0]
                phase = (step % 4)  # Cycle through phases
                print("Phase: " + str(phase))
                # Phases 0-3
                api.set_light(first_light, phase)

    finally:
        # Stop the simulation
        api.stop_simulation()
