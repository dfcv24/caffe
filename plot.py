import matplotlib.pyplot as plt
import sys

list_train_loss = []
list_test_loss = []
f = open(sys.argv[1])
for lines in f.readlines():
	if lines.find('Train net output #0:') != -1:
		train_loss = lines.split()[10]
		#print loss
		list_train_loss.append(train_loss)
	elif lines.find('Test net output #0:') != -1:
		test_loss = lines.split()[10]
		#print test_loss
		list_test_loss.append(test_loss)
f.close()

plot_train_loss = []
count = 0
total_train_loss = float(0)
for x in list_train_loss:
	total_train_loss += float(x)
	if count == 19:
		avg_train_loss = total_train_loss / 20
		total_train_loss = float(0)
		plot_train_loss.append(avg_train_loss)
		count = -1
	count += 1

plot_test_loss = []
count = 0
total_test_loss = float(0)
for x in list_test_loss:
	total_test_loss += float(x)
	if count == 3:
		avg_test_loss = total_test_loss / 4
		total_test_loss = float(0)
		plot_test_loss.append(avg_test_loss)
		count = -1
	count += 1

plot_x = []
for x in xrange(len(plot_train_loss)):
	plot_x.append((x+1)*2000)

plt.xlabel('iterations')
plt.ylabel('loss')
plt.plot(plot_x, plot_test_loss)
plt.title('test_loss')
plt.plot(plot_x, plot_train_loss)
plt.title('train_loss')
plt.grid(True)
plt.show()