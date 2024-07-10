import time
import numpy as np

from mmcv.runner import EpochBasedRunner, get_host_info

from online_src.domain_indicator_orchestrator import DomainIndicator


class OnlineRunner(EpochBasedRunner):
    def __init__(
        self,
        model,
        batch_processor=None,
        optimizer=None,
        work_dir=None,
        logger=None,
        meta=None,
        max_iters=None,
        max_epochs=None,
        source_dataloader=None,
        samples_per_gpu=None,
        domain_indicator_args=None,
        cfg_lr=None,
        mode_train=True,
    ):
        super().__init__(
            model,
            batch_processor,
            optimizer,
            work_dir,
            logger,
            meta,
            max_iters,
            max_epochs,
        )
        self.source_dataloader = source_dataloader
        self.source_iterator = iter(self.source_dataloader)
        self.time_elapsed = []
        self.samples_per_gpu = samples_per_gpu
        self.domain_change = "static.decode_1.loss_seg"
        self.domain_indicator = DomainIndicator(**domain_indicator_args, **cfg_lr)
        self.initial_lr = cfg_lr["initial_lr"]
        self.policy_lr = cfg_lr["policy_lr"]
        self.max_lr = cfg_lr["max_lr"]
        self.lr_far_domain = cfg_lr["lr_far_domain"]
        self.mode_train = mode_train

    def next_source(self):
        try:
            # 获取下一个样本
            source_sample = next(self.source_iterator)
        except StopIteration:
            # 如果迭代器已经遍历完所有样本，则重新初始化迭代器
            self.source_iterator = iter(self.source_dataloader)
            source_sample = next(self.source_iterator)

        return source_sample

    def get_total_fps(self):
        time = np.array(self.time_elapsed)
        # 计算总帧率:用迭代次数乘以每个 GPU 的样本数，然后除以时间总和
        return (self._iter * self.samples_per_gpu) / np.sum(time), np.std(1 / time)

    def get_wandb(self):
        for hook in self.hooks:
            if "Wandb" in hook.__class__.__name__:
                return hook.wandb
        return None

    def get_lr_hook(self):
        for hook in self.hooks:
            if "Lr" in hook.__class__.__name__:
                return hook
        return None

    # 获取学习率迭代次数
    def get_lr_iters(self):
        return self.domain_indicator.iters

    # 获取当前学习率
    def get_lr(self):
        if self.lr_far_domain:
            val = self.domain_indicator.get_domain_value()
            # 设置最小和最大学习率
            min_ratio, max_ratio = self.initial_lr, self.lr_far_domain
            return self.domain_indicator.linear_interpolation(val, [min_ratio, max_ratio], -1)
        else:
            return self.initial_lr

    def get_lr_schedule(self):
        if self.policy_lr == "constant":
            return "constant", self.get_lr() # 策略为constant则直接获取当前学习率
        elif self.policy_lr == "adaptive_init":
            lr = self.domain_indicator.get_lr() # # 获取自适应初始学习率
            cur_lr = self.optimizer.param_groups[0]["lr"] # 获取当前优化器的学习率
            lr = min(lr + cur_lr, self.max_lr) # 确保学习率不超过最大学习率
            lr_config = dict(
                policy="LinearDecay", # 使用线性衰减策略
                min_lr=0.0,  # 最小学习率为 0
                max_progress=self.domain_indicator.base_iters,
                by_epoch=False,
            )
        elif self.policy_lr == "adaptive_slope":
            lr = self.get_lr()
            lr_config = dict(
                policy="LinearDecay",
                min_lr=0.0,
                max_progress=self.domain_indicator.iters_train,
                by_epoch=False,
            )
        else:
            raise ValueError(f"policy lr {self.policy_lr} not valid")

        return lr_config, lr

    def replace_lr_hook(self):
        lr_config, lr = self.get_lr_schedule() # # 获取学习率调度配置和当前学习率

        # set lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        if lr_config == "constant":
            return
        idx = None
        for i, hook in enumerate(self.hooks):
            if "Lr" in hook.__class__.__name__:
                idx = i # 找到学习率 hook 的索引
                break
        if idx is not None:
            del self.hooks[idx] # 删除旧的学习率 hook

            self.register_lr_hook(lr_config) # 注册新的学习率 hook

            for i, hook in enumerate(self.hooks):
                if "Lr" in hook.__class__.__name__:
                    hook.after_selected(self) # 调用新的学习率 hook 的 after_selected 方法

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.mode_train:
            kwargs["domain_indicator"] = self.domain_indicator.get_args() # 获取 domain_indicator 的参数

            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs) # 执行训练步骤
            self.time_elapsed.append(outputs["time"]) # 记录训练时间
        else:
            outputs = self.model.val_step(data_batch, **kwargs) # 执行验证步骤
            self.time_elapsed.append(outputs["time"]) # 记录验证时间
        if not isinstance(outputs, dict):
            raise TypeError(
                '"batch_processor()" or "model.train_step()"'
                'and "model.val_step()" must return a dict'
            )
        if "log_vars" in outputs:
            self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
        self.log_buffer.update(dict(time_speed=outputs["time"]))
        self.outputs = outputs # 保存输出结果
        if self.domain_indicator.domain_indicator:
            self.domain_indicator.add(outputs["log_vars"][self.domain_change]) # 更新域指示器

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader) # 计算最大迭代次数
        self.call_hook("before_train_epoch") # 调用训练前的钩子
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, target_data in enumerate(self.data_loader):
            source_data = self.next_source() # 获取下一个源数据
            data_batch = {
                **source_data,
                "target_img_metas": target_data["img_metas"],
                "target_img": target_data["img"],
            } # 合并源数据和目标数据
            self._inner_iter = i # 设置内部迭代次数
            self.call_hook("before_train_iter") # 调用训练迭代前的钩子
            self.run_iter(data_batch, train_mode=True, **kwargs) # 执行训练迭代

            changed, _ = self.domain_indicator.domain_changed() # 检查域是否改变

            if changed:
                self.replace_lr_hook() # 如果域发生变化，替换学习率 hook

            self.log_buffer.update(
                dict(
                    domain_detected=self.domain_indicator.get_domain(), # 更新检测到的域
                    is_training=self.domain_indicator.is_training() if self.mode_train else False, # 更新是否在训练
                    domain_jump=self.domain_indicator.log_jump, # 更新域跳跃
                    dynamic_dacs=self.domain_indicator.dacs, # 更新动态 DACS
                )
            )
            self.call_hook("after_train_iter") # 调用训练迭代后的钩子
            self._iter += 1 # 迭代次数加 1

        self.call_hook("after_train_epoch") # 调用训练 epoch 后的钩子
        self._epoch += 1 # epoch 次数加 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running."""
        assert isinstance(data_loaders, list) # 确保 data_loaders 是一个列表

        assert self._max_epochs is not None, "max_epochs must be specified during instantiation" # 确保 max_epochs 已经指定

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == "train":
                self._max_iters = self._max_epochs * len(data_loaders[i]) # 计算最大迭代次数
                break

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info("Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.info("workflow: %s, max: %d epochs", workflow, self._max_epochs)
        self.call_hook("before_run") # 调用运行前的钩子
        wandb = self.get_wandb()

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(f'runner has no method named "{mode}" to run an ' "epoch")
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError("mode in workflow must be a str, but got {}".format(type(mode)))

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        if wandb:
            total_fps, std_fps = self.get_total_fps() # 获取总帧率和帧率标准差
            wandb.run.summary["FPS"] = total_fps
            wandb.run.summary["FPS_std"] = std_fps
        # save fps array just in case in work_dirs
        times = np.array(self.time_elapsed)
        np.save(f"{self.work_dir}/fps_array.npy", times)
        self.call_hook("after_run")  # 调用运行后的钩子
