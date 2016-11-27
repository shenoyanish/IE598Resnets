import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_init_line(firstline,augment):
	params=dict()
	params['alpha'] = float(firstline.split(' ')[3].split('(')[1].split(',')[0].split('=')[1])
	if not augment:	
		params['batchsize']= int(firstline.split(' ')[4].split(',')[0].split('=')[1])
		params['epoch']= int(firstline.split(' ')[6].split(',')[0].split('=')[1])
		params['gpu']= int(firstline.split(' ')[7].split(',')[0].split('=')[1])
		params['lr']= float(firstline.split(' ')[8].split(',')[0].split('=')[1])
		params['lr_decay_freq']= float(firstline.split(' ')[9].split(',')[0].split('=')[1])
		params['lr_decay_ratio']= float(firstline.split(' ')[10].split(',')[0].split('=')[1])
		params['models']= firstline.split(' ')[11].split(',')[0].split('=')[1]
		params['opt']= firstline.split(' ')[12].split(',')[0].split('=')[1]
		params['weight_decay'] =float(firstline.split(' ')[16].split(')')[0].split('=')[1])
	else:
		params['batchsize']= int(firstline.split(' ')[5].split(',')[0].split('=')[1])
		params['epoch']= int(firstline.split(' ')[7].split(',')[0].split('=')[1])
		params['gpu']= int(firstline.split(' ')[8].split(',')[0].split('=')[1])
		params['lr']= float(firstline.split(' ')[9].split(',')[0].split('=')[1])
		params['lr_decay_freq']= float(firstline.split(' ')[10].split(',')[0].split('=')[1])
		params['lr_decay_ratio']= float(firstline.split(' ')[11].split(',')[0].split('=')[1])
		params['models']= firstline.split(' ')[12].split(',')[0].split('=')[1]
		params['opt']= firstline.split(' ')[13].split(',')[0].split('=')[1]
		params['weight_decay'] =float(firstline.split(' ')[17].split(')')[0].split('=')[1])

	return params

def parse_output_file(filename, is_augment):
	filecontent = open(filename,'r')
	trainlength=50000
	n_iterations = int(np.ceil(50000/200))
	train_loss = np.zeros((200*n_iterations,1))
	train_accuracy = np.zeros((200*n_iterations,1))
	test_accuracy = np.zeros((200,1))

	testlength=10000
	epoch_finished = False
	epoch_begin = False
	learning_rates = list()
	train_loss = list()
	train_accuracy = list()
	test_accuracy = list()
	# test_accuracy_list = list()
	# train_accuracy_list = list()
	for i,line in enumerate(filecontent):
		if i==0:
			params = parse_init_line(line,is_augment)
			n_iterations = int(np.ceil(50000.0/params['batchsize']))
		else:
			if 'learning rate' in line:
				learning_rates.append(float(line.split(':')[-1]))
			elif '/' in line:
				train_loss.append(float(line.split('\t')[1].split(':')[1]))
				train_accuracy.append(float(line.split('\t')[2].strip().split(':')[1]))
			elif 'test accuracy' in line:
				test_accuracy.append(float(line.split('\t')[1].split(':')[1]))
	
	return params,np.asarray(learning_rates),np.asarray(train_loss),np.asarray(train_accuracy),np.asarray(test_accuracy)

params,learning_rates,train_loss,train_accuracy,test_accuracy = parse_output_file('../log.txt',True)


plt.subplot(2,2,1)
plt.plot(learning_rates)
plt.subplot(2,2,2)
plt.plot(train_loss)
plt.subplot(2,2,3)
plt.plot(1.0-train_accuracy)
plt.subplot(2,2,4)
plt.plot(1.0-test_accuracy)
plt.show()








