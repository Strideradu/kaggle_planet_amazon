## https://github.com/colesbury/examples/blob/8a19c609a43dc7a74d1d4c71efcda102eed59365/imagenet/main.py

from net.common import *
from net.dataset.tool import *
from net.utility.tool import *

from net.dataset.kgforest import *
from sklearn.metrics import fbeta_score

from net.model.pyramidnet import PyNet_10 as Net


## global setting ################
SIZE =  112   #128  #112





##--- helpe functions  -------------
def change_images(images, agument):

    num = len(images)
    if agument == 'left-right' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,1)

    if agument == 'up-down' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,0)

    if agument == 'rotate':
        for n in range(num):
            image = images[n]
            images[n] = randomRotate90(image)  ##randomRotate90  ##randomRotate


    return images


#------------------------------------------------------------------------------------------------------------
#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
def find_f_measure_threshold1(probs, labels, thresholds=None):

    #f0 = fbeta_score(labels, probs, beta=2, average='samples')  #micro  #samples
    def _f_measure(probs, labels, threshold=0.5, beta=2 ):

        SMALL = 1e-12 #0  #1e-12
        batch_size, num_classes = labels.shape[0:2]

        l = labels
        p = probs>threshold

        num_pos     = p.sum(axis=1) + SMALL
        num_pos_hat = l.sum(axis=1)
        tp          = (l*p).sum(axis=1)
        precise     = tp/num_pos
        recall      = tp/num_pos_hat

        fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
        f  = fs.sum()/batch_size
        return f


    best_threshold =  0
    best_score     = -1

    if thresholds is None:
        thresholds = np.arange(0,1,0.005)
        ##thresholds = np.unique(probs)

    N=len(thresholds)
    scores = np.zeros(N,np.float32)
    for n in range(N):
        t = thresholds[n]
        #score = f_measure(probs, labels, threshold=t)
        score = fbeta_score(labels, probs>t, beta=2, average='samples')  #micro  #samples
        scores[n] = score

    return thresholds, scores



## https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
# def find_f_measure_threshold2(probs, labels, resolution=100, seed=0.20):
#     p,y = probs, labels
#     batch_size, num_classes = labels.shape[0:2]
#
#     def mf(x):
#         p2 = np.zeros_like(p)
#         for i in range(17):
#             p2[:, i] = (p[:, i] > x[i]).astype(np.int)
#         score = fbeta_score(y, p2, beta=2, average='samples')
#         return score
#
#     thesholds = [seed]*num_classes
#     scores = [0]*num_classes
#     for i in range(num_classes):
#
#         best_theshold = 0
#         best_score    = 0
#         for theshold in range(1,resolution):
#             theshold /= resolution
#             thesholds[i] = theshold
#             score = mf(thesholds)
#             if score > best_score:
#                 best_theshold = theshold
#                 best_score = score
#
#         thesholds[i] = best_theshold
#         scores[i]    = best_score
#         print('\t(i, best_theshold, best_score)=%2d, %0.3f, %f'%(i, best_theshold, best_score))
#
#     print('')
#     return thesholds, scores



def find_f_measure_threshold2(probs, labels, num_iters=100, seed=0.235):

    batch_size, num_classes = labels.shape[0:2]

    best_thresholds = [seed]*num_classes
    best_scores     = [0]*num_classes
    for t in range(num_classes):

        thresholds = [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if  f2 > best_scores[t]:
                best_scores[t]     = f2
                best_thresholds[t] = th
        print('\t(t, best_thresholds[t], best_scores[t])=%2d, %0.3f, %f'%(t, best_thresholds[t], best_scores[t]))
    print('')
    return best_thresholds, best_scores
#------------------------------------------------------------------------------------------------------------


#precision_recall_curve
def binary_precision_recall_curve(labels, predictions, beta=2):

    precise, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions)
    f2 = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + 1e-12)  #beta=2  #f2 score
    idx = np.argmax(f2)

    return precise, recall, f2, threshold, idx




# write csv
def write_submission_csv(csv_file, predictions, thresholds):

    class_names = CLASS_NAMES
    num_classes = len(class_names)

    with open(KAGGLE_DATA_DIR+'/split/test-61191') as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    num_test = len(names)


    assert((num_test,num_classes) == predictions.shape)
    with open(csv_file,'w') as f:
        f.write('image_name,tags\n')
        for n in range(num_test):
            shortname = names[n].split('/')[-1].replace('.<ext>','')

            prediction = predictions[n]
            s = score_to_class_names(prediction, class_names, threshold=thresholds)
            f.write('%s,%s\n'%(shortname,s))







# loss ----------------------------------------
def multi_criterion(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels))
    return loss

#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
#f  = fbeta_score(labels, probs, beta=2, average='samples')
def multi_f_measure( probs, labels, threshold=0.235, beta=2 ):

    SMALL = 1e-6 #0  #1e-12
    batch_size = probs.size()[0]

    #weather
    l = labels
    p = (probs>threshold).float()

    num_pos     = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp          = torch.sum(l*p,1)
    precise     = tp/(num_pos     + SMALL)
    recall      = tp/(num_pos_hat + SMALL)

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f  = fs.sum()/batch_size
    return f



## main functions ############################################################
def predict(net, test_loader):

    test_dataset = test_loader.dataset
    num_classes  = len(test_dataset.class_names)
    predictions  = np.zeros((test_dataset.num,num_classes),np.float32)

    test_num  = 0
    for iter, (images, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))

        batch_size = len(images)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1,num_classes)

    assert(test_dataset.num==test_num)

    return predictions



def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))
        loss  = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == test_loader.dataset.num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc



def evaluate_and_predict(net, test_loader):

    test_dataset = test_loader.dataset
    num_classes  = len(test_dataset.class_names)
    predictions  = np.zeros((test_dataset.num,num_classes),np.float32)

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))
        loss  = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1,num_classes)

    assert(test_dataset.num==test_num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc, predictions




def do_training():

    out_dir ='/root/share/project/pytorch/results/kaggle-forest/PyNet_10-jpg-baseline-1'

    ## ------------------------------------

    os.makedirs(out_dir +'/snap', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    num_classes = len(CLASS_NAMES)
    batch_size  = 96 #96  #80 #96 #96   #96 #32  #96 #128 #

    train_dataset = KgForestDataset('train-40479',  #'train-32479',#'train-3000',##'train-3000',##
                                    transform=transforms.Compose([
                                        #transforms.Lambda(lambda x: randomRotate(x, u=0.5, limit=45)),
                                        #transforms.Lambda(lambda x: randomShiftScale (x, u=0.5, limit=5)),
                                        transforms.Lambda(lambda x: randomShiftScaleRotate(x, u=0.5, shift_limit=4, scale_limit=4, rotate_limit=45)),
                                        transforms.Lambda(lambda x: randomFlip(x)),
                                        transforms.Lambda(lambda x: randomTranspose(x)),
                                        transforms.Lambda(lambda x: toTensor(x)),
                                    ]),
                                    height=SIZE,width=SIZE,
                                    label_csv='train_v2.csv')
    train_loader  = DataLoader(
                        train_dataset,
                        sampler=RandomSampler(train_dataset),  ##
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 3,
                        pin_memory  = True)

    test_dataset = KgForestDataset('valid-8000',
                                   transform=transforms.Compose([
                                         transforms.Lambda(lambda x: toTensor(x)),
                                   ]),
                                    height=SIZE,width=SIZE,
                                    label_csv='train_v2.csv')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  #None,
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 3,
                        pin_memory  = True)


    height, width , in_channels   = test_dataset.images[0].shape
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    log.write('\ttrain_dataset.num = %d\n'%(train_dataset.num))
    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset.num  = %d\n'%(test_dataset.num))
    log.write('\tbatch_size        = %d\n'%batch_size)
    log.write('\ttrain_loader.sampler = %s\n'%(str(train_loader.sampler)))
    log.write('\ttest_loader.sampler  = %s\n'%(str(test_loader.sampler)))
    log.write('\n')

    # if 0:  ## check data
    #     check_kgforest_dataset(train_dataset, train_loader)
    #     exit(0)


    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net((in_channels,height,width), num_classes)
    ##if initial_model is not None:  net = torch.load(initial_model)

    net.cuda()
    #log.write('\n%s\n'%(str(net)))
    log.write('%s\n\n'%(type(net)))
    log.write(inspect.getsource(net.__init__)+'\n')
    log.write(inspect.getsource(net.forward)+'\n')
    log.write('\n')


    ## optimiser ----------------------------------
    num_epoches = 50  #100
    it_print    = 20
    epoch_test  = 1
    epoch_save  = 5


    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)  ###0.0005
    #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

    ## start training here! ###
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' epoch   iter   rate  |  smooth_loss   |  train_loss  (acc)  |  valid_loss  (acc)  | min\n')
    log.write('----------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss  = np.nan
    train_acc   = np.nan
    test_loss   = np.nan
    test_acc    = np.nan
    time = 0

    start0 = timer()
    for epoch in range(num_epoches):  # loop over the dataset multiple times
        #print ('epoch=%d'%epoch)
        start = timer()

        if 1:
            lr = 0.1 # schduler here ---------------------------
            if epoch>=10: lr=0.010
            if epoch>=25: lr=0.005
            if epoch>=35: lr=0.001
            if epoch>=40: lr=0.0001
            if epoch> 45: break

            adjust_learning_rate(optimizer, lr)

        rate =  get_learning_rate(optimizer)[0] #check
        #--------------------------------------------------------

        sum_smooth_loss = 0.0
        sum = 0
        net.cuda().train()

        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):

            logits, probs = net(Variable(images.cuda()))
            loss  = multi_criterion(logits, labels.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #additional metrics
            sum_smooth_loss += loss.data[0]
            sum += 1

            # print statistics
            if it % it_print == it_print-1:
                smooth_loss = sum_smooth_loss/sum
                sum_smooth_loss = 0.0
                sum = 0

                train_acc  = multi_f_measure(probs.data, labels.cuda())
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.3f  | %0.3f  %5.3f | ... ' % \
                        (epoch + it/num_its, it + 1, rate, smooth_loss, train_loss, train_acc),\
                        end='',flush=True)


        #---- end of one epoch -----
        end = timer()
        time = (end - start)/60

        if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:

            net.cuda().eval()
            test_loss,test_acc = evaluate(net, test_loader)

            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   |  %0.3f  | %0.3f  %5.3f | %0.3f  %5.3f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, test_loss,test_acc, time))

        if epoch % epoch_save == epoch_save-1 or epoch == num_epoches-1:
            torch.save(net,out_dir +'/snap/%03d.torch'%(epoch+1))


    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60

    ## check : load model and re-test
    torch.save(net,out_dir +'/snap/final.torch')
    if 1:
        net = torch.load(out_dir +'/snap/final.torch')

        net.cuda().eval()
        test_loss, test_acc, predictions = evaluate_and_predict( net, test_loader )

        log.write('\n')
        log.write('%s:\n'%(out_dir +'/snap/final.torch'))
        log.write('\tall time to train=%0.1f min\n'%(time0))
        log.write('\ttest_loss=%f, test_acc=%f\n'%(test_loss,test_acc))

        assert(type(test_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
        #dump_results(out_dir +'/train' , predictions, test_dataset.labels, test_dataset.names)




##to determine best threshold etc ... ## -----------------------------------------------------------

def do_find_thresholds():

    out_dir    ='/root/share/project/pytorch/results/kaggle-forest/PyNet_10-jpg-baseline-1'
    model_file = out_dir +'/snap/final.torch'  #final


    ## ------------------------------------
    log = Logger()
    log.open(out_dir+'/log.thresholds.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 128

    test_dataset = KgForestDataset('train-40479',  #'valid-8000',   #'train-simple-road-280',
                                   transform=transforms.Compose([
                                         transforms.Lambda(lambda x: toTensor(x)),
                                   ]),
                                    height=SIZE,width=SIZE,
                                    label_csv='train_v2.csv')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  #None,
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 5,
                        pin_memory  = True)

    height, width , in_channels   = test_dataset.images[0].shape
    log.write('\t(height,width)     = (%d, %d)\n'%(height,width))
    log.write('\tin_channels        = %d\n'%(in_channels))
    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset.num   = %d\n'%(test_dataset.num))
    log.write('\tbatch_size         = %d\n'%batch_size)
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\n')

    net = torch.load(model_file)
    net.cuda().eval()


    # do testing here ###
    aguments = ['default', 'left-right', 'up-down' ,'rotate',] #'default'  # #'up-down'   #'default' #
    num_augments = len(aguments)
    num_classes  = len(test_dataset.class_names)
    test_num     = test_dataset.num
    test_dataset_images = test_dataset.images.copy()

    all_predictions = np.zeros((num_augments+1, test_num, num_classes),np.float32)
    for a in range(num_augments):
        agument = aguments[a]
        log.write('** predict @ agument = %s **\n'%agument)

        test_dataset.images = change_images(test_dataset_images,agument)
        test_loss, test_acc, predictions = evaluate_and_predict( net, test_loader )
        all_predictions[a] = predictions

        log.write('\t\ttest_loss=%f, test_acc=%f\n\n'%(test_loss,test_acc))


    # add average case ...
    aguments = aguments + ['average']
    predictions = all_predictions.sum(axis=0)/num_augments
    all_predictions[num_augments] = predictions
    log.write('\n')


    # find thresholds and save all
    labels = test_dataset.labels
    for a in range(num_augments+1):
        agument = aguments[a]
        predictions = all_predictions[a]

        test_dir = out_dir +'/thresholds/'+ agument
        os.makedirs(test_dir, exist_ok=True)
        log.write('** thresholding @ test_dir = %s **\n'%test_dir)

        #find threshold --------------------------
        if 1:
            thresholds, scores = find_f_measure_threshold1(predictions, labels)
            i = np.argmax(scores)
            best_threshold, best_score = thresholds[i], scores[i]

            log.write('\tmethod1:\n')
            log.write('\tbest_threshold=%f, best_score=%f\n\n'%(best_threshold, best_score))
            #plot_f_measure_threshold(thresholds, scores)
            #plt.pause(0)

        if 1:
            seed = best_threshold  #0.21  #best_threshold

            best_thresholds,  best_scores = find_f_measure_threshold2(predictions, labels, num_iters=100, seed=0.21)
            log.write('\tmethod2:\n')
            log.write('\tbest_threshold\n')
            log.write (str(best_thresholds)+'\n')
            log.write('\tbest_scores\n')
            log.write (str(best_scores)+'\n')
            log.write('\n')

        #--------------------------------------------
        assert(type(test_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
        np.save(test_dir +'/predictions.npy',predictions)
        np.save(test_dir +'/labels.npy',labels)
        np.savetxt(test_dir +'/best_threshold.txt', np.array(best_thresholds),fmt='%.5f' )
        np.savetxt(test_dir +'/best_scores.txt',np.array(best_scores),fmt='%.5f')
    # pass


##-----------------------------------------
def do_submissions():


    out_dir    ='/root/share/project/pytorch/results/kaggle-forest/PyNet_10-jpg-baseline-1'
    model_file = out_dir +'/snap/final.torch'  #final

    #modify here!
    thresholds=\
        [0.25, 0.19, 0.23, 0.13, 0.31, 0.25, 0.2, 0.21, 0.2, 0.28, 0.07, 0.09, 0.15, 0.16, 0.09, 0.12, 0.02]
        #[0.21, 0.18, 0.26, 0.13, 0.3, 0.24, 0.2, 0.23, 0.2, 0.29, 0.07, 0.07, 0.13, 0.1, 0.09, 0.12, 0.03]


    thresholds=np.array(thresholds)
    ## ------------------------------------

    log = Logger()
    log.open(out_dir+'/log.submissions.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size    = 128

    test_dataset = KgForestDataset('test-61191',  #'valid-8000',   #'train-simple-road-280',
                                   transform=transforms.Compose([
                                         transforms.Lambda(lambda x: toTensor(x)),
                                   ]),
                                    height=SIZE,width=SIZE,
                                    label_csv=None)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  #None,
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 5,
                        pin_memory  = True)

    height, width , in_channels   = test_dataset.images[0].shape
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset.num  = %d\n'%(test_dataset.num))
    log.write('\tbatch_size        = %d\n'%batch_size)
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\n')

    net = torch.load(model_file)
    net.cuda().eval()


    # do testing here ###
    aguments = ['default', 'left-right', 'up-down' ,'rotate',] #'default'  # #'up-down'   #'default' #
    num_augments = len(aguments)
    num_classes  = len(test_dataset.class_names)
    test_num     = test_dataset.num
    test_dataset_images = test_dataset.images.copy()

    all_predictions = np.zeros((num_augments+1,test_num,num_classes),np.float32)
    for a in range(num_augments):
        agument = aguments[a]
        log.write('** predict @ agument = %s **\n'%agument)

        ## perturb here for test argumnetation  ## ----
        test_dataset.images = change_images(test_dataset_images,agument)
        predictions = predict( net, test_loader )
        all_predictions[a] = predictions

    # add average case ...
    aguments = aguments + ['average']
    predictions = all_predictions.sum(axis=0)/num_augments
    all_predictions[num_augments] = predictions
    log.write('\n')

    # apply thresholds and save all
    for a in range(num_augments+1):
        agument = aguments[a]
        predictions = all_predictions[a]

        test_dir = out_dir +'/submissions/'+ agument
        os.makedirs(test_dir, exist_ok=True)


        assert(type(test_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
        np.save(test_dir +'/predictions.npy',predictions)
        np.savetxt(test_dir +'/thresholds.txt',thresholds)
        write_submission_csv(test_dir + '/results.csv', predictions, thresholds )

    pass




# averaging over many models ####################################################################
def do_find_thresholds1():

    out_dir    ='/root/share/project/pytorch/results/kaggle-forest/PyNet_4-all-1'
    model_file = out_dir +'/snap/final.torch'  #final


    ## ------------------------------------
    log = Logger()
    log.open(out_dir+'/log.thresholds.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')



    predict_files = [
        '/root/share/project/pytorch/results/kaggle-forest/MultiNet-02a-jpg/thresholds/average6/predictions.npy',
        '/root/share/project/pytorch/results/kaggle-forest/pynet-18a-jpg/thresholds/average5/predictions.npy',
        '/root/share/project/pytorch/results/kaggle-forest/PyNet_4-all-1/thresholds/average/predictions.npy',
    ]
    label_file='/root/share/project/pytorch/results/kaggle-forest/PyNet_4-all-1/thresholds/average/labels.npy'

    # averaging -----------
    num = len(predict_files)
    predictions = np.load(predict_files[0])
    for n in range(1,num):
        predictions += np.load(predict_files[n])
    predictions = predictions/num

    labels = np.load(label_file)



    test_dir = out_dir +'/thresholds/average11'
    os.makedirs(test_dir, exist_ok=True)

    #find threshold --------------------------
    if 1:
        log.write('\tmethod1:\n')

        thresholds, scores = find_f_measure_threshold1(predictions, labels)
        i = np.argmax(scores)
        best_threshold, best_score = thresholds[i], scores[i]

        log.write('\tbest_threshold=%f, best_score=%f\n\n'%(best_threshold, best_score))
        #plot_f_measure_threshold(thresholds, scores)
        #plt.pause(0)

    if 1:
        log.write('\tmethod2:\n')

        seed = best_threshold  #0.21  #best_threshold
        best_thresholds,  best_scores = find_f_measure_threshold2(predictions, labels, num_iters=100, seed=seed)

        log.write('\tbest_threshold\n')
        log.write (str(best_thresholds)+'\n')
        log.write('\tbest_scores\n')
        log.write (str(best_scores)+'\n')
        log.write('\n')

    #--------------------------------------------
    np.save(test_dir +'/predictions.npy',predictions)
    np.save(test_dir +'/labels.npy',labels)
    np.savetxt(test_dir +'/best_threshold.txt', np.array(best_thresholds),fmt='%.5f' )
    np.savetxt(test_dir +'/best_scores.txt',np.array(best_scores),fmt='%.5f')

    # pass




#averaging over many models
def do_submission1():

    out_dir ='/root/share/project/pytorch/results/kaggle-forest/PyNet_4-all-1'
    thresholds = \
        [0.24, 0.2, 0.23, 0.12, 0.32, 0.22, 0.2, 0.21, 0.25, 0.26, 0.07, 0.07, 0.11, 0.12, 0.08, 0.09, 0.03]

    thresholds = (np.array(thresholds)+0.01)
    ## ------------
    log = Logger()
    log.open(out_dir+'/log.submission.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')


    predict_files = [
        '/root/share/project/pytorch/results/kaggle-forest/MultiNet-02a-jpg/submissions/average2/predictions.npy',
        '/root/share/project/pytorch/results/kaggle-forest/pynet-18a-jpg/submissions/average2a/predictions.npy',
        '/root/share/project/pytorch/results/kaggle-forest/PyNet_4-all-1/submissions/average/predictions.npy',
        #'/root/share/project/pytorch/results/kaggle-forest/pynet-18a-jpg/submissions/default/predictions.npy',
        #'/root/share/project/pytorch/results/kaggle-forest/pynet-18a-jpg/submissions/left-right/predictions.npy',
        #'/root/share/project/pytorch/results/kaggle-forest/pynet-18a-jpg/submissions/up-down/predictions.npy',
        #'/root/share/project/pytorch/results/kaggle-forest/pynet-18a-jpg/submissions/rotate/predictions.npy',
   ]

    # averaging -----------
    num=len(predict_files)
    predictions = np.load(predict_files[0])
    for n in range(1,num):
        predictions += np.load(predict_files[n])
    predictions = predictions/num
    # -------------------


    test_dir = out_dir +'/submissions/average12'
    os.makedirs(test_dir, exist_ok=True)

    write_submission_csv(test_dir + '/results.csv', predictions, thresholds )
    np.save(test_dir +'/predictions.npy',predictions)
    np.savetxt(test_dir +'/thresholds.txt',thresholds)

    with open(test_dir +'/predict_files.txt', 'w') as f:
        for file in predict_files:
            f.write('%s\n'%file)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #do_training()
    #do_find_thresholds()
    do_submissions()


    #do_find_thresholds1()
    #do_submission1()

    print('\nsucess!')
