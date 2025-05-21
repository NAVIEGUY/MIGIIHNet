# RRDB
nf = 3
gc = 32

# Super parameters
clamp = 2.0
channels_in = 3
log10_lr1 = -5.0
log10_lr2 = -5.0
lr1 = 2.0 * 10 ** log10_lr1
lr2 = 2.0 * 10 ** log10_lr2
epochs = 5000
weight_decay = 1e-5
init_scale = 0.01

device_ids = [0]
pre = False
# Super loss
save_image = './test_pictures/sample_8'
MODEL_PATH = './weighting path/checkpoint_3'

# Train:
batch_size = 2
val_batch_size = 2
mea_batch_size = 2
cropsize = 256
betas1 = (0.5, 0.999)
betas2 = (0.5, 0.999)
weight_step = 200
gamma = 0.98
val_freq = 1
# Val:
# cropsize_val_coco = 256
# cropsize_val_imagenet = 256
cropsize_val_div2k = 256
cropsize_mea_div2k = 256
