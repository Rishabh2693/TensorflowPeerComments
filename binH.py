from __future__ import division
import tarfile
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import re
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import random

df = pd.read_csv("/home/rsinha2/data2/comments_27_3.csv")
#temporary
df = df.drop(range(97,198),axis=0)

headers = list(df.columns.values)
headers.remove('Unnamed: 0')
headers.remove('Unnamed: 9')
headers.remove('Unnamed: 10')
headers.remove('Unnamed: 11')
headers.remove('Unnamed: 12')

#temporary
df = df[df.Praise.notnull()]
df = df[df.Problem.notnull()]
df = df[df.Mitigation.notnull()]
df = df[df.Summary.notnull()]
df = df[df.Solution.notnull()]
df = df[df.Neutrality.notnull()]
df = df[df.Localization.notnull()]

df_x = df.loc[:, ['Comments']]

headers.remove('Comments')
headers = ["Neutrality"]

df_y = df.loc[:, headers]
df_y.head()
df_y[df_y != 0] = 1
df_y = df_y.round(0).astype(int)
df_y['new'] = 1 - df_y
#load model
model = Doc2Vec.load(os.path.join("trained", "comments2vec.d2v"))


comments = []
for index, row in df.iterrows():
    line = row["Comments"]
    line = re.sub("[^a-zA-Z?!]"," ", line)
    words = [w.lower().decode('utf-8') for w in line.strip().split() if len(w)>=3]
    comments.append(words)
x_train = []
for comment in comments:
        feature_vec = model.infer_vector(comment)
        #feature_vec = np.append(feature_vec,len(comment))
        x_train.append(feature_vec)


x_test = x_train[len(x_train)-100:len(x_train)]
x_train = x_train[0:len(x_train)-100]

#x_test = []
#for i in range(100,110):
#    x_test.append(model.docvecs["COMMENT_"+str(i)])

#y_train = y[0:100]
y_train = df_y[0:len(x_train)]
#y_test = y[100:110]
y_test = df_y[len(x_train):]
inputX = np.array(x_train)
inputY = y_train.as_matrix()
outputX = np.array(x_test)
outputY = y_test.as_matrix()
numFeatures = inputX.shape[1]
numLabels = 2
numEpochs  = 25000
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=inputX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])


weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6/numFeatures+
                                                         numLabels+1)),
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=(np.sqrt(6/numFeatures+numLabels+1)),
                                    name="bias"))

init_OP = tf.global_variables_initializer()

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

epoch_values=[]
accuracy_values=[]
cost_values=[]
# Turn on interactive plotting
plt.ion()
# Create the main, super plot
#fig = plt.figure()
# Create two subplots on their own axes and give titles
#ax1 = plt.subplot("211")
#ax1.set_title("TRAINING ACCURACY", fontsize=18)
#ax2 = plt.subplot("212")
#ax2.set_title("TRAINING COST", fontsize=18)
#plt.tight_layout()


sess = tf.Session()

# Initialize all tensorflow variables
sess.run(init_OP)

## Ops for vizualization
# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
# False is 0 and True is 1, what was our average?
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)
# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
# Merge all summaries
all_summary_OPS = tf.summary.merge_all()
# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)

# Initialize reporting variables
cost = 0
diff = 1

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: inputX, yGold: inputY})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            summary_results, train_accuracy, newCost = sess.run(
                [all_summary_OPS, accuracy_OP, cost_OP], 
                feed_dict={X: inputX, yGold: inputY}
            )
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Write summary stats to writer
            writer.add_summary(summary_results, i)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, cost %g"%(i, newCost))
            print("step %d, change in cost %g"%(i, diff))

            # Plot progress to our two subplots
 #           accuracyLine, = ax1.plot(epoch_values, accuracy_values)
 #           costLine, = ax2.plot(epoch_values, cost_values)
 #           fig.canvas.draw()
#            time.sleep(1)


# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 
                                                     feed_dict={X: outputX, 
                                                                yGold: outputY})))

saver = tf.train.Saver()
# Save variables to .ckpt file
# saver.save(sess, "trained_variables.ckpt")

# Close tensorflow session
sess.close()
