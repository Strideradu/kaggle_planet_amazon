from net.common import *
from net.dataset.tool import *
import pandas as pd


# helper functions -------------
def score_to_class_names(prob, class_names, threshold = 0.5, nil=''):

    N = len(class_names)
    if not isinstance(threshold,(list, tuple, np.ndarray)) : threshold = [threshold]*N

    s=nil
    for n in range(N):
        if prob[n]>threshold[n]:
            if s==nil:
                s = class_names[n]
            else:
                s = '%s %s'%(s, class_names[n])
    return s


def draw_class_names(image,  prob, class_names, threshold=0.5):

    weather = CLASS_NAMES[:4]
    s = score_to_class_names(prob, class_names, threshold, nil=' ')
    for i, ss in enumerate(s.split(' ')):
        if ss in weather:
            color = (255,255,0)
        else:
            color = (0, 255,255)

        draw_shadow_text(image, ' '+ss, (5,30+(i)*15),  0.5, color, 1)



def create_image(image, width=256, height=256):
    h,w,c = image.shape

    if c==3:
        jpg_src=0
        tif_src=None

        M=1
        jpg_dst=0

    if c==4:
        jpg_src=None
        tif_src=0

        M=2
        tif_dst=0

    if c==7:
        jpg_src=0
        tif_src=3

        M=3
        jpg_dst=0
        tif_dst=1


    img = np.zeros((h,w*M,3),np.uint8)
    if jpg_src is not None:
        jpg_blue  = image[:,:,jpg_src  ] *255
        jpg_green = image[:,:,jpg_src+1] *255
        jpg_red   = image[:,:,jpg_src+2] *255

        img[:,jpg_dst*w:(jpg_dst+1)*w] = np.dstack((jpg_blue,jpg_green,jpg_red)).astype(np.uint8)

    if tif_src is not None:
        tif_blue  = np.clip(image[:,:,tif_src  ] *4095*255/65536.0*6 -25-30,a_min=0,a_max=255)
        tif_green = np.clip(image[:,:,tif_src+1] *4095*255/65536.0*6    -30,a_min=0,a_max=255)
        tif_red   = np.clip(image[:,:,tif_src+2] *4095*255/65536.0*6 +25-30,a_min=0,a_max=255)
        tif_nir   = np.clip(image[:,:,tif_src+3] *4095*255/65536.0*4,a_min=0,a_max=255)

        img[:,tif_dst*w:(tif_dst+1)*w] = np.dstack((tif_blue,tif_green,tif_red)).astype(np.uint8)
        img[:,tif_dst(+1)*w: ] = np.dstack((tif_nir,tif_nir,tif_nir)).astype(np.uint8)



    if height!=h or width!=w:
        img = cv2.resize(img,(width*M,height))

    return img






## custom data loader -----------------------------------
class KgForestDataset(Dataset):

    def __init__(self, split, transform=None, height=64, width=64, label_csv='train_v2.csv'):
        data_dir    = KAGGLE_DATA_DIR
        class_names = CLASS_NAMES
        num_classes = len(class_names)

        # read names
        list = data_dir +'/split/'+ split
        with open(list) as f:
            names = f.readlines()
        names = [x.strip()for x in names]
        num   = len(names)


        #read images
        images = np.zeros((num,height,width,3),dtype=np.float32)
        for n in range(num):
            img_file  = data_dir + '/image/' + names[n]
            jpg_file  = img_file.replace('<ext>','jpg')
            tif_file  = img_file.replace('<ext>','tif')
            image_jpg = cv2.imread(jpg_file,1)
            #image_tif = io.imread(tif_file)

            h,w = image_jpg.shape[0:2]
            if height!=h or width!=w:
                image_jpg = cv2.resize(image_jpg,(height,width))
                #image_tif = cv2.resize(image_tif,(height,width))

                #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
                #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

            images[n,:,:,:3]=image_jpg.astype(np.float32)/255.0
            #images[n,:,:,3:]=image_tif.astype(np.float32)/4095.0  #2^12=4096

            if 0: #debug
                image = create_image(images[n])
                im_show('image',image,1)
                cv2.waitKey(0)
            pass


        #read labels
        df     = None
        labels = None
        if label_csv is not None:
            labels = np.zeros((num,num_classes),dtype=np.float32)

            csv_file  = data_dir + '/image/' + label_csv   # read all annotations
            df = pd.read_csv(csv_file)
            for c in class_names:
                df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)

            df1 = df.set_index('image_name')
            for n in range(num):
                shortname = names[n].split('/')[-1].replace('.<ext>','')
                labels[n] = df1.loc[shortname].values[1:]

                if 0: #debug
                    image = create_image(images[n])
                    draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                    draw_class_names(image, labels[n], class_names)
                    im_show('image', image)
                    cv2.waitKey(0)

                    #images[n]=cv2.resize(image,(height,width)) ##mark for debug
                    pass
        #save
        self.transform = transform
        self.num       = num
        self.split     = split
        self.names     = names
        self.images    = images

        self.class_names = class_names
        self.df     = df
        self.labels = labels


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)

        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img, index

        else:
            label = self.labels[index]
            return img, label, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.images)




def check_kgforest_dataset(dataset, loader):

    class_names = dataset.class_names
    names  = dataset.names

    if dataset.labels is not None:
        for i, (images, labels, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            # get the inputs
            num = len(images)
            for n in range(num):
                label = labels[n].numpy()
                image = tensor_to_img(images[n], mean=0, std=1, dtype=np.float32)

                s = score_to_class_names(label, class_names)
                print('%32s : %s %s'%  (names[indices[n]], label.T, s))

                image = create_image(image)
                shortname = names[indices[n]].split('/')[-1].replace('.<ext>','')
                draw_shadow_text(image, shortname, (5,15),  0.5, (255,255,255), 1)
                draw_class_names(image, label, class_names)
                im_show('image',image)
                cv2.waitKey(1)
                #print('\t\tlabel=%d : %s'%(label,classes[label]))
                #print('')

    if dataset.labels is None:
        for i, (images, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            # get the inputs
            num = len(images)
            for n in range(num):
                image = tensor_to_img(images[n], mean=0, std=1, dtype=np.float32)

                print('%32s : nil'% (names[indices[n]]))

                image = create_image(image)
                shortname = names[indices[n]].split('/')[-1].replace('.<ext>','')
                draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                im_show('image',image)
                cv2.waitKey(1)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



    dataset = KgForestDataset('valid-8000', ##'train-40479',  ##'train-ordered-20', ##
                                transforms.Compose([

                                    #transforms.Lambda(lambda x: randomRotate(x, u=0.5, limit=45)),
                                    #transforms.Lambda(lambda x: randomShift (x, u=0.5, limit=5)),
                                    #transforms.Lambda(lambda x: randomShiftAScale (x, u=0.5, limit=5)),
                                    transforms.Lambda(lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=6, scale_limit=6, rotate_limit=45)),
                                    transforms.Lambda(lambda x: randomFlip(x)),
                                    transforms.Lambda(lambda x: randomTranspose(x)),
                                    transforms.Lambda(lambda x: toTensor(x)),
                                ]),
                                #label_csv=None,
                              )
    sampler = FixedSampler(dataset,[5,]*100  )  #SequentialSampler(dataset)  #SequentialSampler  #RandomSampler
    loader  = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2, drop_last=False, pin_memory=True)


    for epoch in range(100):
        print('epoch=%d -------------------------'%(epoch))
        check_kgforest_dataset(dataset, loader)

    print('sucess')