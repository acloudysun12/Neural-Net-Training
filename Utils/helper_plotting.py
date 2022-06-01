import numpy as np
import random
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    return(plt.tight_layout())


def plot_accuracy(train_acc, valid_acc):

    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs+1), 
             train_acc, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()


def plot_generated_images(data_loader, model, device, 
                          unnormalizer=None,
                          figsize=(20, 2.5), n_images=15, modeltype='autoencoder'):

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (features, _) in enumerate(data_loader):
        
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        
        with torch.no_grad():
            if modeltype == 'autoencoder':
                decoded_images = model(features)[:n_images]
            elif modeltype == 'VAE':
                encoded, z_mean, z_log_var, decoded_images = model(features)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')



def plot_only_real_gen(orig_images, decoded_images, n_images = 10, figsize=(20, 4), unnormalizer = None): 
    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                         sharex=True, sharey=True, figsize=figsize)
    
    color_channels = orig_images.shape[1]
    image_height = orig_images.shape[2]
    image_width = orig_images.shape[3]
    
    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')    

                

def plot_only_gen(decoded_images, n_images = 10, figsize=(20, 4), unnormalizer = None):
    
    color_channels = decoded_images.shape[1]
    image_height = decoded_images.shape[2]
    image_width = decoded_images.shape[3]
    
    fig, axes = plt.subplots(nrows=1, ncols=n_images, 
                         sharex=True, sharey=True, figsize=figsize)

    # MAKE THE PLOTS
    #     for i in range(n_images):
    for i, ax in enumerate(axes):
        curr_img = decoded_images[i,:,:,:].detach().to(torch.device('cpu'))        
        if unnormalizer is not None:
            curr_img = unnormalizer(curr_img)

        if color_channels > 1:
            curr_img = np.transpose(curr_img, (1, 2, 0))
            ax.imshow(curr_img)
        else:
            ax.imshow(curr_img.view((image_height, image_width)), cmap='binary')
                
                
                
# +
def plot_generated_images_new(data_loader, model, device, 
                              unnormalizer=None,
                              figsize=(20, 6), n_images=5, 
                              plot_num = None, modeltype='autoencoder', fixed_idx = False):

    fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    if fixed_idx == False:
        rand_idx = np.random.randint(1, int(len(data_loader)/4)) # SAMPLE FROM LESS BATCHES FOR FASTER SPEED
        iter_perf = iter(data_loader)
        for numiters in range(rand_idx):
            perf_sample = next(iter_perf)
    else:
        torch.manual_seed(20935)
        np.random.seed(20935)
        random.seed(20935)
        rand_idx = 3 # THIS NUMBER IS PICKED ARBITRARILY
        iter_perf = iter(data_loader)
        for numiters in range(rand_idx):
            perf_sample = next(iter_perf)
    
    
    ### RUN MODEL AND GET DECODED SAMPLE IMAGES
    model.eval()
    
    labels = perf_sample[1]
    features = perf_sample[0]

    color_channels = features.shape[1]
    image_height = features.shape[2]
    image_width = features.shape[3]
    
    # ADD IN PLOTTING NUMBERS
    if plot_num is not None:
        labels = labels == plot_num
        features = features[labels,:,:,:]

    
    orig_images = features[:n_images].detach().to(torch.device('cpu'))
    print(orig_images.shape)
    
    with torch.no_grad():
        if modeltype == 'autoencoder':
            decoded_images = model(orig_images)
        elif modeltype == 'VAE':
            encoded, z_mean, z_log_var, decoded_images = model(orig_images)
        else:
            raise ValueError('`modeltype` not supported')
    
#     all_images = torch.cat((orig_images,decoded_images), 0).shape
    
    # MAKE THE PLOTS
    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')


# -

def plot_brand_new_images(data_loader, model, device, 
                          figsize = (20, 6), n_images = 10,
                          unnormalizer = None, 
                          n_sample = -1, # represents how many values to use for calculating the meanmean/meanvar
                          plot_num = 1):

    fig, axes = plt.subplots(nrows=1, ncols=n_images, 
                             sharex=True, sharey=True, figsize=figsize)
    
    rand_idx = np.random.randint(1, int(len(data_loader)/4)) # SAMPLE FROM LESS BATCHES FOR FASTER SPEED
    iter_perf = iter(data_loader)
    for numiters in range(rand_idx):
        perf_sample = next(iter_perf)
    
    
    ### RUN MODEL AND GET DECODED SAMPLE IMAGES
    model.eval()
    
    labels = perf_sample[1]
    features = perf_sample[0]

    color_channels = features.shape[1]
    image_height = features.shape[2]
    image_width = features.shape[3]
    
    # ADD IN PLOTTING NUMBERS
    # if plot_num is None --> basically will just return the "average-looking" number
    if plot_num is not None:
        labels = labels == plot_num
        features = features[labels,:,:,:]
    
    print('Used ' + str(features.shape) + ' observations to get the latent mean / var')
    
    if n_sample == -1:
        n_sample = features.shape[0]
    
    orig_images = features[:n_sample].detach().to(torch.device('cpu'))
    
    encoded, z_mean, z_log_var, decoded_images = model(orig_images)
        
    # Get the MEAN of the z_means. This way we generate from ONE mean
    meanmean = torch.mean(z_mean, dim = 0)
    # Get the SD of the mean of the exp(log_vars) (aka sqrt(mean(exp(z_logvar))))
    meansd = torch.sqrt(torch.mean(torch.exp(z_log_var), dim = 0))
    
    # Generate a bunch of obs of Z ~ N(0,1), where  
    eps = torch.randn(100, meanmean.size(0)).to('cpu')
    # Generate X ~ u + sig*Z
    newgen_imgs = meanmean + eps * meansd
    # Decode the images with our model
    decoded_images = model.decoder(newgen_imgs)
    # We only want to plot plot how many images
    print('Generated ' + str(decoded_images.shape) + ' new observations')
    decoded_images = decoded_images[:n_images,:,:,:]

    # MAKE THE PLOTS
#     for i in range(n_images):
    for i, ax in enumerate(axes):
        curr_img = decoded_images[i,:,:,:].detach().to(torch.device('cpu'))        
        if unnormalizer is not None:
            curr_img = unnormalizer(curr_img)

        if color_channels > 1:
            curr_img = np.transpose(curr_img, (1, 2, 0))
            ax.imshow(curr_img)
        else:
            ax.imshow(curr_img.view((image_height, image_width)), cmap='binary')


def plot_latent_space_with_labels(num_classes, data_loader, encoding_fn, device, FIGSIZE = (10, 5), dim1 = 0, dim2 = 1):
    d = {i:[] for i in range(num_classes)}

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)
            
            embedding = encoding_fn(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    
    plt.figure(figsize = FIGSIZE) # SET THE FIGURE SIZE, OUTSIDE THE LOOP
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, dim1], d[i][:, dim2],
            color=colors[i][1],
            label=f'{i}',
            alpha=0.5)

    plt.legend()


def plot_latent_space_with_labels_AE(num_classes, data_loader, model, device, FIGSIZE = (10, 5)):
    d = {i:[] for i in range(num_classes)}

    model.eval()
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)
            
            embedding = model.encoder(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    
    plt.figure(figsize = FIGSIZE) # SET THE FIGURE SIZE, OUTSIDE THE LOOP
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, 0], d[i][:, 1],
            color=colors[i][1],
            label=f'{i}',
            alpha=0.5)

    plt.legend()


def plot_images_sampled_from_vae(model, device, latent_size, unnormalizer=None, num_images=10):

    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################    

        rand_features = torch.randn(num_images, latent_size).to(device)
        new_images = model.decoder(rand_features)
        color_channels = new_images.shape[1]
        image_height = new_images.shape[2]
        image_width = new_images.shape[3]

        ##########################
        ### VISUALIZATION
        ##########################

        image_width = 28

        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
        decoded_images = new_images[:num_images]

        for ax, img in zip(axes, decoded_images):
            curr_img = img.detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax.imshow(curr_img)
            else:
                ax.imshow(curr_img.view((image_height, image_width)), cmap='binary') 


def plot_modified_faces(original, diff,
                        diff_coefficients=(0., 0.5, 1., 1.5, 2., 2.5, 3.),
                        decoding_fn=None,
                        device=None,
                        figsize=(8, 2.5)):

    fig, axes = plt.subplots(nrows=2, ncols=len(diff_coefficients), 
                             sharex=True, sharey=True, figsize=figsize)
    

    for i, alpha in enumerate(diff_coefficients):
        more = original + alpha*diff
        less = original - alpha*diff
        
        
        if decoding_fn is not None:
            ######################################
            ### Latent -> Original space
            with torch.no_grad():

                if device is not None:
                    more = more.to(device).unsqueeze(0)
                    less = less.to(device).unsqueeze(0)

                more = decoding_fn(more).to('cpu').squeeze(0)
                less = decoding_fn(less).to('cpu').squeeze(0)
            ###################################### 
        
        if not alpha:
            s = 'original'
        else:
            s = f'$\\alpha=${alpha}'
            
        axes[0][i].set_title(s)
        axes[0][i].imshow(more.permute(1, 2, 0))
        axes[1][i].imshow(less.permute(1, 2, 0))
        axes[1][i].axison = False
        axes[0][i].axison = False



def plot_modified_faces_v2(original, diff,
                          diff_coefficients=(0., 0.5, 1., 1.5, 2., 2.5, 3.),
                          decoding_fn=None,
                          device=None,
                          figsize=(8, 2.5)):

    fig, axes = plt.subplots(nrows=2, ncols=len(diff_coefficients), 
                             sharex=True, sharey=True, figsize=figsize)
    

    for i, alpha in enumerate(diff_coefficients):
        more = original + alpha*diff
        less = original - alpha*diff
        
      
        if not alpha:
            s = 'original'
        else:
            s = f'$\\alpha=${alpha}'
            
        axes[0][i].set_title(s)
        axes[0][i].imshow(more.squeeze().permute(1, 2, 0).detach().numpy())
        axes[1][i].imshow(less.squeeze().permute(1, 2, 0).detach().numpy())
        axes[1][i].axison = False
        axes[0][i].axison = False



def plot_training_loss_v2(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    return(plt)
