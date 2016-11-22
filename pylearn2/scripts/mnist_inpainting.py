"""
Adaptation of reconstruct_mnist_gif.py from https://github.com/laurent-dinh/nice/tree/master/pylearn2/scripts
to inpaint MNIST dataset. To use it, set up NICE repository and make sure everything is working properly 
according to authors' instructions. Then run this script instead of reconstruct_mnist_gif.py

Usage:
minst_inpainting.py path/to/model/nice_mnist_best.pkl path/to/corrupted/dataset

Notes:
- nice_mnist_best.pkl must be trained beforehand calling python pylearn2/scripts/train.py exp/nice_mnist.yaml 
(train is in the pylearn2 lib, not NICE lib)
- Corrupted dataset must contain index.txt and index_mask.txt file containing mask info. The supplied experiment is 
set to inpaint the MNIST validation set.
- Takes about 4 hours to inpaint 10k mnist samples. Duration can be controlled by setting the sqrt_iter variable.
In the original code this was set to 70, but this was too long and the improvement not significant enough
to justify the extra time.
"""

import numpy as np
import theano
import theano.tensor as T
from pylearn2.utils import serial
from theano.compat.python2x import OrderedDict
import matplotlib
matplotlib.use('Agg')

from pylearn2.utils import sharedX
from pylearn2.config import yaml_parse
from pylearn2.datasets.mnist import MNIST
import os, sys
import os.path
from os.path import join
import PIL.Image
import progressbar
from pylearn2.utils.image import Image, ensure_Image

def save_single_image(x, (h,w), save_dir, save_name ): 
    # Squeeze values into [0,1] since the input is in the range [-1,1]
    xc = x*0.5 
    xc += 0.5
    assert xc.min() >= 0.0
    assert xc.max() <= 1.0

    xc = np.cast['uint8'](xc * 255.0)
    img = Image.fromarray(xc.reshape((h,w)))

    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    img.save(join(save_dir,save_name))


def load_image(path, scale=255.0):
    return np.float32(PIL.Image.open(path)) / scale

def load_imgs_from_dir(imgs_dir, img_size, index_fname):
    if not os.path.exists(imgs_dir):
        raise IOError('Error- %s doesn\'t exist!' % imgs_dir)
    raw_imgs_data = np.loadtxt(os.path.join(imgs_dir,index_fname),delimiter=' ',dtype=str)
    total_images = raw_imgs_data.shape[0]
    imgs = np.zeros((total_images,img_size))
    for idx in np.arange(total_images):
        imgs[idx,:] = load_image(os.path.join(imgs_dir,raw_imgs_data[idx][0])).reshape(img_size)
    return imgs.astype('float32')

# Loading model
_, model_path, db_dir = sys.argv
model = serial.load(model_path)
print "Model loaded"


# Defining the number of examples
cols = 10
rows = 10

n_examples = rows * cols
batch_size = n_examples
input_dim = 784
spatial_dims = (28,28)
X_shared_val = np.random.uniform(size=(n_examples, input_dim))
X_shared = sharedX(X_shared_val, 'X_shared')
print "Initialize"


# Create the iterative reconstruction function
X = T.matrix('X')
M = T.imatrix('M')

X_complete = T.where(M, X, X_shared)
ll = model.get_log_likelihood(X_complete)

grad = T.grad(ll.mean(), X_shared, disconnected_inputs='warn')
updates = OrderedDict()

lr = T.scalar('lr')
is_noise = sharedX(0., 'is_noise')
updates[X_shared] = X_shared + lr * (grad + model.prior.theano_rng.normal(size=X_shared.shape))
updates[X_shared] = T.where(M, X, updates[X_shared])

f = theano.function([X, M, lr], [ll.mean()], updates=updates, allow_input_downcast=True)
print 'Compiled training function'

# Setup for training and display
dataset_yaml_src = model.dataset_yaml_src
train_set = yaml_parse.load(dataset_yaml_src)
dataset = train_set
num_samples = n_examples

vis_batch = dataset.get_batch_topo(num_samples)
rval = tuple(vis_batch.shape[dataset.X_topo_space.axes.index(axis)]
             for axis in ('b', 0, 1, 'c'))
_, patch_rows, patch_cols, channels = rval

# Get examples and masks


raw_im_data = np.loadtxt(os.path.join(db_dir,'index.txt'),delimiter=' ',dtype=str)
N = raw_im_data.shape[0]
# flip mask to match the code convention (0 is masked pixel)
masks = 1 - load_imgs_from_dir(db_dir,input_dim,'index_mask.txt') 
inpaint_imgs = load_imgs_from_dir(db_dir,input_dim,'index.txt')
n_batches = N / batch_size

new_db_path = db_dir + '_nice_ip'
if not os.path.exists(new_db_path):
    os.mkdir(new_db_path)
    
pbar = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('\rProcessed %(value)d of %(max)d Batches '), progressbar.Bar()], maxval=n_batches, term_width=50).start()
with open(join(new_db_path,'index.txt'),'wb') as db_file:

    for b in np.arange(n_batches):
        X_shared.set_value(np.random.uniform(size=(n_examples, input_dim)).astype('float32'))
        x_val = inpaint_imgs[b*batch_size:(b+1)*batch_size,:]
        m_val = masks[b*batch_size:(b+1)*batch_size,:]
        X_shared.set_value(np.where(m_val == 1, x_val, X_shared.get_value()))
        sqrt_iter = 30 # originally was 70. 
        max_iter = np.arange(sqrt_iter)**2
        max_iter = max_iter.sum()

        iteration = 0
        for i in xrange(sqrt_iter):
            for j in xrange(i**2):
                rval = f(x_val, m_val, 1/(.1*iteration+10))[0]
                iteration += 1
                
        vis_batch_subset = X_shared.get_value()[:(rows * cols)]
        display_batch = dataset.adjust_for_viewer(vis_batch_subset)
        for idx in np.arange(batch_size):
            
            abs_idx = (b*batch_size)+idx
            save_name = raw_im_data[abs_idx][0].replace('corrupted','ip')
            save_single_image(display_batch[idx,:], spatial_dims, new_db_path,save_name)
            db_file.write('%s %s\n' % ( save_name, raw_im_data[abs_idx][1]))
        pbar.update(b)
        
    pbar.finish()

