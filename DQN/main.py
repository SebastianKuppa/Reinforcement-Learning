import agent
import time

if __name__ == "__main__":
    TstartTime = time.time()
    # perform model grid search for DQN
    agent.grid_search()
    TendTime = time.time()
    print(f'Training took {round((TendTime - TstartTime) / 60)} Minutes')
    print(f'Training took {round((TendTime - TstartTime) / 3600)} Hours')
