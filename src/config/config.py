def get_cfg():
    """
    Get a copy of the default config.
    """
    from .defaults import _C
    return _C.clone()


# def imagedata_args(cfg):
#     return {
#         'root': cfg.data.root,
#         'sources': cfg.data.sources,
#         'targets': cfg.data.targets,
#         'height': cfg.data.height,
#         'width': cfg.data.width,
#         'transforms': cfg.data.transforms,
#         'norm_mean': cfg.data.norm_mean,
#         'norm_std': cfg.data.norm_std,
#         'use_gpu': cfg.use_gpu,
#         'split_id': cfg.data.split_id,
#         'combineall': cfg.data.combineall,
#         'load_train_targets': cfg.data.load_train_targets,
#         'batch_size_train': cfg.train.batch_size,
#         'batch_size_test': cfg.test.batch_size,
#         'workers': cfg.data.workers,
#         'num_instances': cfg.sampler.num_instances,
#         'train_sampler': cfg.sampler.train_sampler,
#         # image
#         'cuhk03_labeled': cfg.cuhk03.labeled_images,
#         'cuhk03_classic_split': cfg.cuhk03.classic_split,
#         'market1501_500k': cfg.market1501.use_500k_distractors,
#     }
#
#
# def videodata_args(cfg):
#     return {
#         'root': cfg.data.root,
#         'sources': cfg.data.sources,
#         'targets': cfg.data.targets,
#         'height': cfg.data.height,
#         'width': cfg.data.width,
#         'transforms': cfg.data.transforms,
#         'norm_mean': cfg.data.norm_mean,
#         'norm_std': cfg.data.norm_std,
#         'use_gpu': cfg.use_gpu,
#         'split_id': cfg.data.split_id,
#         'combineall': cfg.data.combineall,
#         'batch_size_train': cfg.train.batch_size,
#         'batch_size_test': cfg.test.batch_size,
#         'workers': cfg.data.workers,
#         'num_instances': cfg.sampler.num_instances,
#         'train_sampler': cfg.sampler.train_sampler,
#         # video
#         'seq_len': cfg.video.seq_len,
#         'sample_method': cfg.video.sample_method
#     }
#
#
# def optimizer_args(cfg):
#     return {
#         'optim': cfg.train.optim,
#         'lr': cfg.train.lr,
#         'weight_decay': cfg.train.weight_decay,
#         'momentum': cfg.sgd.momentum,
#         'sgd_dampening': cfg.sgd.dampening,
#         'sgd_nesterov': cfg.sgd.nesterov,
#         'rmsprop_alpha': cfg.rmsprop.alpha,
#         'adam_beta1': cfg.adam.beta1,
#         'adam_beta2': cfg.adam.beta2,
#         'staged_lr': cfg.train.staged_lr,
#         'new_layers': cfg.train.new_layers,
#         'base_lr_mult': cfg.train.base_lr_mult
#     }
#
#
# def lr_scheduler_args(cfg):
#     return {
#         'lr_scheduler': cfg.train.lr_scheduler,
#         'stepsize': cfg.train.stepsize,
#         'gamma': cfg.train.gamma,
#         'max_epoch': cfg.train.max_epoch
#     }
#
#
# def engine_run_args(cfg):
#     return {
#         'save_dir': cfg.data.save_dir,
#         'max_epoch': cfg.train.max_epoch,
#         'start_epoch': cfg.train.start_epoch,
#         'fixbase_epoch': cfg.train.fixbase_epoch,
#         'open_layers': cfg.train.open_layers,
#         'start_eval': cfg.test.start_eval,
#         'eval_freq': cfg.test.eval_freq,
#         'test_only': cfg.test.evaluate,
#         'print_freq': cfg.train.print_freq,
#         'dist_metric': cfg.test.dist_metric,
#         'normalize_feature': cfg.test.normalize_feature,
#         'visrank': cfg.test.visrank,
#         'visrank_topk': cfg.test.visrank_topk,
#         'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
#         'ranks': cfg.test.ranks,
#         'rerank': cfg.test.rerank
#     }
