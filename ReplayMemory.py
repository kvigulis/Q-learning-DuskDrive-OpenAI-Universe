import numpy as np




# train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

stateArray = np.zeros([1,102,160,2], dtype='uint8')
q_valueArray = np.zeros([1,8])
actionArray = np.zeros([1])
rewardArray = np.zeros([1])

end_episode = False
def clearReplayMemory():
    global stateArray
    global q_valueArray
    global actionArray
    global rewardArray

    stateArray = np.zeros([1,102,160,2], dtype='uint8')
    q_valueArray = np.zeros([1,8])
    actionArray = np.zeros([1])
    rewardArray = np.zeros([1])

def backSweep(discount_factor = 0.994):

    for k in reversed(range(len(actionArray) - 1)):
        # Get the data for the k'th state in the replay-memory.
        action = int(actionArray.item(k))
        print "action k : ", action
        reward = rewardArray.item(k)

        action_value = reward + discount_factor * np.max(q_valueArray[k + 1])

        q_valueArray[k, action] = action_value

    return None

def shuffleStatesAndQValuesInUnison():

    rngState = np.random.get_state()
    np.random.shuffle(stateArray)
    np.random.set_state(rngState)
    np.random.shuffle(q_valueArray)
