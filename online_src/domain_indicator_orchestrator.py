import numpy as np


class DomainIndicator:
    def __init__(
        self,
        domain_indicator,
        limit,
        threshold_src,
        threshold,
        base_iters,
        dynamic_dacs,
        max_iters,
        initial_lr,
        policy_lr,
        max_lr,
        threshold_max,
        reduce_training,
        far_domain,
        lr_far_domain,
    ):
        self.domain_indicator = domain_indicator
        self.limit = limit
        self.mem = []
        self.threshold_src = threshold_src
        self.threshold_max = threshold_max
        self.threshold_up, self.threshold_down = threshold
        self.domain = 0
        self.domains = []
        self.prev = None
        self.losses = []
        self.dynamic_dacs = dynamic_dacs
        self.dacs = 0.5 if not self.dynamic_dacs else self.dynamic_dacs[0]
        self.iters = self.iters_train = 0
        self.base_iters = base_iters
        self.log_jump = 0
        self.MAX_ITERS_TRAIN = max_iters
        self.initial_lr = initial_lr
        self.policy_lr = policy_lr
        self.lr = initial_lr
        self.max_lr = max_lr
        self.lr_far_domain = lr_far_domain
        self.min_reduce, self.max_reduce = reduce_training
        self.far_domain = far_domain
        self.last_domain = None

    # 计算 self.mem 列表中所有元素的平均值是否小于等于 threshold_src
    def _is_source(self):
        return self.avg() <= self.threshold_src

    # 判断当前的域值是否属于“远域”（即是否大于或等于 self.far_domain)
    def is_far_domain(self):
        val = self.get_domain_value()
        return val >= self.far_domain

    def get_domain_value(self):
        return self.avg() if self.mem else self.last_domain

    def linear_interpolation(self, x, vals, dire):
        return (
            # 线性插值,x是插值点,vals是插值区间
            np.interp(x, [self.threshold_src, self.threshold_max], vals)
            if dire < 0
            else 1
        )

    def get_domain(self):
        return self.domain

    def get_lr(self):
        return self.lr

    def add(self, val):
        self.mem.append(val)

    def avg(self):
        return np.mean(self.mem)

    # 学习率根据当前值 val 和前一个值 self.prev 之间的差值进行调整
    # 值越大，学习率调整的幅度越大
    def _calculate_lr(self, val):
        diff = abs(val - self.prev)
        ratio = diff / self.threshold_up
        return min(2 * ratio * self.initial_lr, self.max_lr)

    def _jump(self, old, new):
        diff = new - old
        # 判断当前值 new 是否大于旧值 old 的上阈值 self.threshold_up 或小于旧值 old 的下阈值 self.threshold_down
        return diff > self.threshold_up or diff < self.threshold_down

    # 计算训练迭代次数
    def _calculate_iters_train(self, val, iters):
        diff = abs(val - self.prev)
        ratio = diff / self.threshold_up
        return int(ratio * iters)

    def _update_args(self, changed, dire, val):
        if not changed:
            return
        # if we are going to a higher intensity use less source in DACS
        if self.dynamic_dacs:
            min_ratio, max_ratio = self.dynamic_dacs
            # 根据 val 计算 dacs 参数
            self.dacs = self.linear_interpolation(val, [min_ratio, max_ratio], -1)

        # 根据 val 和 dire 计算新的迭代次数 iters
        iters = self.base_iters * self.linear_interpolation(
            val, [self.min_reduce, self.max_reduce], dire
        )

        # 计算新的训练迭代次数 self.iters_train，并确保不超过最大值 self.MAX_ITERS_TRAIN
        if self.policy_lr != "adaptive_init":
            self.iters_train = self._calculate_iters_train(val, iters) + max(
                self.iters_train - self.iters, 0
            )
            self.iters_train = min(self.iters_train, self.MAX_ITERS_TRAIN)
        else:
            # 计算新的学习率 self.lr，并更新训练迭代次数
            self.lr = self._calculate_lr(val)
            self.iters_train = iters
        self.iters = 0

    # 检测领域是否发生变化
    def domain_changed(self):
        self.log_jump = 0
        # 如果内存长度小于限制长度或者没有域指示器，则不进行域变化检测
        if len(self.mem) < self.limit or not self.domain_indicator:
            return False, None

        val = self.avg()
        self.losses.append(val)

        # 如果前一个值 self.prev 为 None，则将当前值 val 赋值给 self.prev
        if self.prev is None:
            self.prev = val
            return False, None

        changed, dire = False, None
        # 如果当前值 val 与前一个值 self.prev 之间的差值大于上阈值 self.threshold_up 或小于下阈值 self.threshold_down，则认为发生了跳跃
        if self._jump(self.prev, val):
            changed, dire = True, np.sign(val - self.prev)
            self._update_args(changed, dire, val)
            self.log_jump = self.prev - val
            self.prev = None
            self.domain += dire

        # 重置 mem 和 last_domain
        self.mem = []
        self.last_domain = val
        return changed, dire

    def is_training(self):
        if not self.domain_indicator:
            return True
        return self.iters < self.iters_train

    def get_args(self):
        if not self.domain_indicator:
            return {}

        train = self.is_training()
        # 增加迭代计数器
        self.iters += 1
        return dict(
            dacs=self.dacs,
            train=train,
        )
