def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=10, max_iter=30000, power=0.9):
    if iter % lr_decay_iter or iter > max_iter:
        return None

    new_lr = init_lr * (1 - iter / max_iter)**power
    optimizer.params_groups[0]['lr'] = new_lr
    optimizer.params_groups[1]['lr'] = 10 * new_lr