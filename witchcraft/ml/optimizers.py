import tensorflow as tf

class Optimizer:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    # pylint: disable=R0201
    def to_tf_optimizer(self) -> tf.train.Optimizer:
        return tf.train.GradientDescentOptimizer(learning_rate=1.0)

class WitchcraftAdamOptimizer(Optimizer):
    def __init__(self,
                 learning_rate: float = 1.0,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8
                 ) -> None:
        Optimizer.__init__(self, "adam")
        self._learning_rate: float = learning_rate
        self._beta1: float = beta1
        self._beta2: float = beta2
        self._epsilon: float = epsilon

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def get_beta_1(self) -> float:
        return self._beta1

    def get_beta_2(self) -> float:
        return self._beta2

    def get_epsilon(self) -> float:
        return self._epsilon

    def to_tf_optimizer(self) -> tf.train.Optimizer:
        return tf.train.AdamOptimizer(
            learning_rate=self.get_learning_rate(),
            beta1=self.get_beta_1(),
            beta2=self.get_beta_2(),
            epsilon=self.get_epsilon(),
        )

class WitchcraftAdagradOptimizer(Optimizer):
    def __init__(self,
                 learning_rate: float = 1.0,
                 initial_accumulator_value: float = 0.1
                 ) -> None:
        Optimizer.__init__(self, "adagrad")
        self._learning_rate: float = learning_rate
        self._initial_accumulator_value: float = initial_accumulator_value

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def get_initial_accumulator_value(self) -> float:
        return self._initial_accumulator_value

    def to_tf_optimizer(self) -> tf.train.Optimizer:
        return tf.train.AdagradOptimizer(
            learning_rate=self.get_learning_rate(),
            initial_accumulator_value=self.get_initial_accumulator_value()
        )

class WitchcraftGradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 1.0) -> None:
        Optimizer.__init__(self, "gd")
        self._learning_rate: float = learning_rate

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def to_tf_optimizer(self) -> tf.train.Optimizer:
        return tf.train.GradientDescentOptimizer(
            learning_rate=self.get_learning_rate()
        )

class WitchcraftMomentumOptimizer(Optimizer):
    def __init__(self, learning_rate: float = 1.0, momentum: float = 0.9) -> None:
        Optimizer.__init__(self, "momentum")
        self._learning_rate: float = learning_rate
        self._momentum: float = momentum

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def get_momentum(self) -> float:
        return self._momentum

    def to_tf_optimizer(self) -> tf.train.Optimizer:
        return tf.train.MomentumOptimizer(
            learning_rate=self.get_learning_rate(),
            momentum=self.get_momentum()
        )
