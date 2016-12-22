import matplotlib.pyplot as plt
import sys

def average_list(list_name, num):
	count = 0
	total_item = float(0)
	plot_list = []
	for x in list_name:
		total_item += float(x)
		if count == num - 1:
			avg_item = total_item / num
			total_item = float(0)
			plot_list.append(avg_item)
			count = -1
		count += 1
	return plot_list

list_train_iden_accu = []
list_train_iden_p_accu = []
list_train_vari_accu = []
list_train_iden_loss = []
list_train_iden_p_loss = []
list_train_vari_loss = []
list_test_vari_loss = []
list_test_vari_accu = []
f = open(sys.argv[1])
# 0 1 2 3 4 5 2 5
for lines in f.readlines():
	if lines.find('Train net output #0:') != -1:
		train_accu = lines.split()[10]
		list_train_iden_accu.append(train_accu)
	elif lines.find('Train net output #3:') != -1:
		train_accu = lines.split()[10]
		list_train_iden_p_accu.append(train_accu)
	elif lines.find('Train net output #4:') != -1:
		train_accu = lines.split()[10]
		list_train_vari_accu.append(train_accu)
	elif lines.find('Train net output #1:') != -1:
		train_loss = lines.split()[10]
		list_train_iden_loss.append(train_loss)
	elif lines.find('Train net output #2:') != -1:
		train_loss = lines.split()[10]
		list_train_iden_p_loss.append(train_loss)
	elif lines.find('Train net output #5:') != -1:
		train_loss = lines.split()[10]
		list_train_vari_loss.append(train_loss)	
	elif lines.find('Test net output #4:') != -1:
		test_accu = lines.split()[10]
		list_test_vari_accu.append(test_accu)
	elif lines.find('Test net output #5:') != -1:
		test_loss = lines.split()[10]
		list_test_vari_loss.append(test_loss)
f.close()

plot_train_iden_accu = average_list(list_train_iden_accu, 5)
plot_train_iden_p_accu = average_list(list_train_iden_p_accu, 5)
plot_train_vari_accu = average_list(list_train_vari_accu, 5)
plot_train_iden_loss = average_list(list_train_iden_loss, 5)
plot_train_iden_p_loss = average_list(list_train_iden_p_loss, 5)
plot_train_accu_loss = average_list(list_train_vari_loss, 5)
#print plot_train_iden_p_accu,plot_train_vari_accu,list_test_vari_accu

plot_x = []
for x in xrange(len(plot_train_iden_accu)):
	plot_x.append((x)*500)
plot_y = []
for y in xrange(len(list_test_vari_accu)):
	plot_y.append((y)*500)

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.plot(plot_x, plot_train_iden_accu,label="train_identification_accuracy")
plt.plot(plot_x, plot_train_iden_p_accu,label="train_identification_p_accuracy")
plt.plot(plot_x, plot_train_vari_accu,label="train_varification_accuracy")
plt.plot(plot_y, list_test_vari_accu,label="test_varification_accuracy")
plt.grid(True)
plt.legend(loc=0, borderaxespad=0.)  

plt.subplot(1,2,2)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.plot(plot_x, plot_train_iden_loss, label="train_identification_loss")
plt.plot(plot_x, plot_train_iden_p_loss, label="train_identification_p_loss")
plt.plot(plot_x, plot_train_accu_loss, label="train_varification_loss")
plt.plot(plot_y, list_test_vari_loss, label="test_varification_loss")
plt.grid(True)
plt.legend(loc=1, borderaxespad=0.)
plt.show()