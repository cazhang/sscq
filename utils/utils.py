import torch
import torch.nn.functional as F


def soft_quantization(embedding, codebooks, N_books, tau_sq):
    """Soft quantization"""
    L_word = int(codebooks.size()[1]/N_books)
    x = torch.split(embedding, L_word, dim=1)
    c = torch.split(codebooks, L_word, dim=1)
    #import ipdb; ipdb.set_trace()
    for i in range(N_books):
        soft_c = F.softmax(-1*squared_distances(x[i], c[i])/tau_sq, dim=-1)
        sim_p = torch.matmul(F.normalize(x[i], dim=1), F.normalize(c[i], dim=1).transpose(1,0))
        if i==0:
            z = soft_c @ c[i]
            p = sim_p
        else:
            z = torch.cat((z, soft_c @ c[i]), dim=1)
            p = torch.cat((p, sim_p), dim=0)

    return z, p


def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sum(diff * diff, -1)


def indexing(codebooks, N_books, X):
    """Create quantization index database with database samples and codebooks"""
    l1, l2 = codebooks.shape
    L_word = int(l2/N_books)
    x = torch.split(X, L_word, 1)
    y = torch.split(codebooks, L_word, 1)
    for i in range(N_books):
        diff = squared_distances(x[i], y[i])
        arg = torch.argmin(diff, dim=1)
        min_idx = torch.reshape(arg, [-1, 1])
        if i == 0:
            quant_idx = min_idx
        else:
            quant_idx = torch.cat((quant_idx, min_idx), dim=1)
    return quant_idx


def pqDist_one(codebooks, N_books, g_x, q_x, device=torch.device('cuda'), book_idx=None):
    """Compute distance between query and database samples"""
    #import ipdb; ipdb.set_trace()
    l1, l2 = codebooks.shape # 16 x 64 (N_words, N_books x L_word)
    L_word = int(l2/N_books) # 16 
    D_codebooks = torch.zeros((l1, N_books), dtype=torch.float32).to(device)

    q_x_split = torch.split(q_x, L_word, 0)
    g_x_split = torch.split(g_x, int(g_x.size(1)/N_books), 1)
    codebooks_split = torch.split(codebooks, L_word, 1)
    D_codebooks_split = torch.split(D_codebooks, 1, 1)

    if book_idx is not None: # using one book
        #print(f'looking at book {book_idx}')
        assert book_idx >= 0 and book_idx < N_books
        for k in range(l1):
            D_codebooks_split[book_idx][k] = torch.norm(q_x_split[book_idx]-codebooks_split[book_idx][k], 2).to(device).detach()
        dist = D_codebooks_split[book_idx][g_x_split[book_idx].long()].to(device)
    else:
        for j in range(N_books): # using all books
            for k in range(l1):
                D_codebooks_split[j][k] = torch.norm(q_x_split[j]-codebooks_split[j][k], 2).to(device).detach()
            if j == 0:
                dist = D_codebooks_split[j][g_x_split[j].long()].to(device)
            else:
                dist = torch.add(dist, D_codebooks_split[j][g_x_split[j].long()].to(device))
    Dpq = torch.squeeze(dist)
    return Dpq.to(device)


