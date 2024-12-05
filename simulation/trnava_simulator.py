import os
import traci


def run_sumo_simulation():
    # Define paths
    sumo_binary = "sumo-gui"  # Use "sumo" for CLI mode
    config_file = os.path.expanduser("~/SmarTTeam/SmarTT/data/sumo_network/osm.sumocfg")

    # Check if the config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        return

    # Start SUMO with TraCI
    try:
        # Start SUMO with TraCI
        traci.start([sumo_binary, "-c", config_file])
        print("SUMO simulation started with TraCI.")

        # Retrieve Traffic Light IDs
        all_tls_ids = traci.trafficlight.getIDList()
        print(f"All Traffic Light IDs in the simulation: {all_tls_ids}")
        # Test each provided ID
        for tls_id in all_tls_ids:
            if tls_id in all_tls_ids:
                print(f"Traffic Light ID '{tls_id}' exists.")
                # Check if it controls lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                if controlled_lanes:
                    print(f"Traffic Light ID '{tls_id}' is controllable. Controlled lanes: {controlled_lanes}")
                else:
                    print(f"Traffic Light ID '{tls_id}' exists but does not control any lanes.")
            else:
                print(f"Traffic Light ID '{tls_id}' does not exist in the simulation.")
        # Close the simulation
        traci.close()
        print("SUMO simulation finished successfully.")
    except Exception as e:
        print(f"Error running SUMO simulation: {e}")


if __name__ == "__main__":
    run_sumo_simulation()
