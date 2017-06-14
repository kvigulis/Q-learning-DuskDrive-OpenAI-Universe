import gym
import universe  # register the universe environments
import numpy as np
import tensorflow as tf
import ImageProcessing
from PIL import Image
import NeuralNetwork as NN
import EpsilonGreedy
import ReplayMemory as RM


sess = tf.InteractiveSession()



actionSpace = [[], #0 For 'No Operation' action. I. e. do nothing.
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)], #1 Forward
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', True)],  #2 Forward-Nitros
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)],  #3 Forward-left
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'N', False)],  #4 Forward-right
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)], #5 Brake
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)],  #6 Brake-left
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'N', False)]]  #7 Brake-right
batch_size = 50
learning_rate = 0.000085

screenTopLeftCorner = [84,18] # Top left corner position. 84 from top, 18 from left margin.
env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()
counter = 0
saveCounter = 0
epsilon = 0.75
state = np.empty([1,102,160,2], dtype='uint8')
Q_values_est = np.empty([1,8])
action = np.random.randint(low=0, high=len(actionSpace))
action_as_1D_array = np.array(action)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, '/home/carl/PycharmProjects/OpenAIUniverseRL/train')

while True:
    action_n = [actionSpace[:][action] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)
    rewardNormalized = np.array([reward_n[0]/40]) # to make the max reward closer to 1.0
    if observation_n[0]:
        pixels_raw = observation_n[0].get("vision")[84:592, 18:818] # Size 508x800. Will be resized by 0.1
        grayscaleImg = ImageProcessing.pre_process_image(pixels_raw) # Makes a 51x80 'uint8' list
        counter += 1
        if counter == 1:
            motionTracer = ImageProcessing.MotionTracer(pixels_raw) # create the object
        else:
            motionTracer.process(pixels_raw) # do stuff to create motion image
            if counter > 399: # Ignore the unearned reward while the car still has it's initial momentum at the start
                # Add data to the replay memory
                # Corresponding Action and Reward gets added one env.step latter than state and q_values
                RM.stateArray = np.vstack((RM.stateArray, state))  # 'state' comes from previous step
                RM.q_valueArray = np.concatenate((RM.q_valueArray, Q_values_est), axis=0)  # 'Q_values_est' comes from previous step
                RM.actionArray = np.concatenate((RM.actionArray, action_as_1D_array), axis=0)
                RM.rewardArray = np.concatenate((RM.rewardArray, rewardNormalized), axis=0)
        # state = Returs 'grayscale' channel [0] and 'grayscale-motion trace' channel [1]
        state_raw = motionTracer.get_state() # Size 102x160x2 :
        state = np.reshape(state_raw, (1,102,160,2)) # 'batch' containing one entry for single estimation.
        state_4D = np.divide(state.astype(np.float32), 50)
        Q_values_est = NN.Q_values_est.eval(feed_dict={NN.x_state: state_4D, NN.keep_prob: 1.0})
        if np.random.random() < epsilon:
            # Choose a random action.
            action = np.random.randint(low=0, high=len(actionSpace))
        else:
            action = np.where(Q_values_est[0]==Q_values_est[0].max())[0][0] # Index of the max Q value
        action_as_1D_array = np.array([action])
        print "Run Number: ", saveCounter
        print "Learning Rate: ", "{:10.14f}".format(learning_rate)
        print "Epsilon: ", epsilon
        print "Action", action_as_1D_array
        print "Reward: ", "{:10.3f}".format(rewardNormalized.item(0)), "Steps: ", counter
        #print "Neural Network: ", Q_values_est
        print "NNet Q Values - NOOP:", "{:10.14f}".format(Q_values_est.item(0)), "; U:", "{:10.14f}".format(Q_values_est.item(1)), "; U+N:", "{:10.14f}".format(Q_values_est.item(2)), "; U+L:", "{:10.14f}".format(Q_values_est.item(3)), "; U+R:", "{:10.14f}".format(Q_values_est.item(4)), "; D:", "{:10.14f}".format(Q_values_est.item(5)), "; D+L:", "{:10.14f}".format(Q_values_est.item(6)), "; D+R:", "{:10.14f}".format(Q_values_est.item(7))

        if counter == 2250:
            env.reset()
            if epsilon > 0.05:
                epsilon = epsilon*0.9996
            else:
                epsilon = 0.05
            learning_rate = learning_rate*0.9998
            print "RM.stateArray.shape: ", RM.stateArray.shape
            print "RM.q_valueArray.shape: ", RM.q_valueArray.shape
            print "RM.actionArray.shape: ", RM.actionArray.shape
            print "RM.rewardArray.shape: ", RM.rewardArray.shape

            print "actions",  RM.actionArray
            print "rewards", RM.rewardArray
            print "q_valueArray old: ", RM.q_valueArray
            RM.backSweep()
            print "q_valueArray new: ", RM.q_valueArray
            #print "\n"
            print "stateArray", RM.stateArray[0][45][12], RM.stateArray[1][45][12], RM.stateArray[2][45][12], RM.stateArray[3][45][12], RM.stateArray[4][45][12]
            #print "q_valueArray new: ", RM.q_valueArray
            RM.shuffleStatesAndQValuesInUnison()
            print "stateArray shuffled", RM.stateArray[0][45][12], RM.stateArray[1][45][12], RM.stateArray[2][45][12], RM.stateArray[3][45][12], RM.stateArray[4][45][12]
            print "q_valueArray shuffled: ", RM.q_valueArray
            for i in range((counter-400)/batch_size):
                print "i:", i

                state_batch = batch = np.divide(RM.stateArray[i*batch_size:(i+1)*batch_size].astype(np.float32), 50)
                Q_value_batch = batch = RM.q_valueArray[i*batch_size:(i+1)*batch_size]
                #print "stateArray", state_batch
                loss = NN.loss.eval(feed_dict={NN.x_state: state_batch, NN.Q_values_new: Q_value_batch, NN.keep_prob: 1.0})
                print "Training run: ", i, "Loss Before: ", loss
                NN.optimizer.run(feed_dict={NN.x_state: state_batch, NN.Q_values_new: Q_value_batch, NN.keep_prob: 1.0, NN.learning_rate: learning_rate})
                loss = NN.loss.eval(feed_dict={NN.x_state: state_batch, NN.Q_values_new: Q_value_batch, NN.keep_prob: 1.0})
                print "Training run: ", i, "Loss After: ", loss
            RM.clearReplayMemory()
            print "Epsilon: ", epsilon
            print "Learning Rate: ",learning_rate
            counter = 0
            saveCounter += 1
            print "Training run: ", saveCounter

            imGray = Image.fromarray(grayscaleImg, mode='L')
            imGray.save('GrayScaleMedium.png')
            imMotion = Image.fromarray(state_raw[:, :, 1], mode='L')
            imMotion.save('MotionSmallMedium.png')

            if saveCounter != 0 and saveCounter%10 == 0:
                saver.save(sess, '/home/carl/PycharmProjects/OpenAIUniverseRL/train')

    env.render()








