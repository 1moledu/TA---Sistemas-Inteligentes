import sys
import os
import time




## importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

from shared_instances import SharedInstances

def main(data_folder_name):

   
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    
    # Instantiate the environment
    env = Env(data_folder)
    
    # config files for the agents
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")
    explorer_file = os.path.join(data_folder, "explorer_config.txt")
    
    # Instantiate agents rescuer and explorer
    SharedInstances.resc4 = Rescuer(env, rescuer_file)
    SharedInstances.resc1 = Rescuer(env, rescuer_file)
    SharedInstances.resc2 = Rescuer(env, rescuer_file)
    SharedInstances.resc3 = Rescuer(env, rescuer_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    SharedInstances.exp = Explorer(env, explorer_file, SharedInstances.resc1, 1, True)
    SharedInstances.exp1 = Explorer(env, explorer_file, SharedInstances.resc2, 3, False)
    SharedInstances.exp2 = Explorer(env, explorer_file, SharedInstances.resc3,5, False)
    SharedInstances.exp3 = Explorer(env, explorer_file, SharedInstances.resc4,7, False)

    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    

    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_408v_94x94")
    
        
    main(data_folder_name)


