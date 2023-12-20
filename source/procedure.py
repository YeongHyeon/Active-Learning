import os
import numpy as np
import source.utils as utils

def training(agent, dataset, epochs, batch_size, sub_k, n_stage):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    dic_history = {'stage':[], 'n_trained':[], 'accuracy':[]}
    iteration = 0
    for stage in range(n_stage):
        print("\nStage %d" %(stage))
        dic_history['stage'].append(stage)
        dic_history['n_trained'].append(dataset.num_trained)

        list_accuracy = []
        for epoch in range(epochs):
            while(True):
                minibatch = dataset.next_batch(batch_size=batch_size, ttvu=0)

                if(minibatch['x'].shape[0] == 0): break
                step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
                iteration += 1

                if(minibatch['terminate']): break

            print("Epoch [%d / %d] (%d iteration)  Loss: %.5f, Acc: %.5f" \
                %(epoch, epochs, iteration, step_dict['losses']['opt'], step_dict['accuracy']))
            list_accuracy.append(step_dict['accuracy'])

        dic_history['accuracy'].append(np.average(list_accuracy))
        agent.save_params(model='stage_%d' %(stage))

        list_loss_unlabel = []
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttvu=3)

            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            list_loss_unlabel.extend(list(step_dict['l_hat']))
            iteration += 1

            if(minibatch['terminate']): break

        dataset.set_by_oracle(loss=list_loss_unlabel, K=sub_k)

    utils.save_pkl(path='training_history.pkl', pkl=dic_history)
    agent.save_params(model='final_epoch')

def test(agent, dataset):

    list_model = utils.sorted_list(os.path.join(agent.path_ckpt, '*.pth'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    best_f1 = 0
    for idx_model, name_model in enumerate(list_model):
        try: agent.load_params(model=name_model)
        except: print("Parameter loading was failed")
        else: print("Parameter loaded")

        print("\nTest... (w/ %s)" %(name_model))

        confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
        while(True):

            minibatch = dataset.next_batch(batch_size=1, ttvu=1)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, training=False)
            y_te = minibatch['y']

            label, logit = np.argmax(y_te[0]), np.argmax(step_dict['score'])
            confusion_matrix[label, logit] += 1

            if(minibatch['terminate']): break

        print("\nConfusion Matrix")
        print(confusion_matrix)

        tot_precision, tot_recall, tot_f1score = 0, 0, 0
        diagonal = 0
        for idx_c in range(dataset.num_class):
            precision = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[:, idx_c])
            recall = confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[idx_c, :])
            f1socre = 2 * (precision * recall / (precision + recall))

            tot_precision += precision
            tot_recall += recall
            tot_f1score += f1socre
            diagonal += confusion_matrix[idx_c, idx_c]
            print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
                %(idx_c, precision, recall, f1socre))

        accuracy = diagonal / np.sum(confusion_matrix)
        print("\nTotal | Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
            %(accuracy, tot_precision/dataset.num_class, tot_recall/dataset.num_class, tot_f1score/dataset.num_class))

        best_f1 = max(best_f1, tot_f1score)

    return best_f1, len(list_model)