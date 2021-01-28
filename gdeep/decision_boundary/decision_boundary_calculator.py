# Joint with Matthias Kemper

import torch
import torch.optim

from typing import Callable

from abc import ABC, abstractmethod

from torchdiffeq import odeint


class DecisionBoundaryCalculator(ABC):
    """
    Abstract class for calculating the decision boundary of
    a neural network
    """
    @abstractmethod
    def step(self, number_of_steps=1):
        """ Performs a single step towards the decision boundary
        """
        pass

    @abstractmethod
    def get_decision_boundary(self) -> torch.Tensor:
        """Return current state and does not make a step

        Returns:
            torch.tensor: current estimate of the decision boundary
        """
        pass

    @abstractmethod
    def get_filtered_decision_boundary(self, delta=0.01) -> torch.Tensor:
        pass

    def _convert_to_distance_to_boundary(self, model, output_shape):
        """Convert the binary or multiclass classifier to a scalar valued model with
        distance to the decision boundary as output.
        """
        if len(output_shape) == 1:
            new_model = lambda x: torch.abs(model(x) - 0.5)
        elif output_shape[-1] == 1:
            new_model = lambda x: torch.abs(model(x).reshape((-1)) - 0.5)
        elif output_shape[-1] == 2:
            new_model = lambda x: torch.abs(model(x)[:, 0] - 0.5)
        else:
            def new_model(x):
                y = torch.topk(model(x), 2).values
                return y[:, 0] - y[:, 1]

        return new_model


class GradientFlowDecisionBoundaryCalculator(DecisionBoundaryCalculator):
    """
    Computes Decision Boundary using the gradient flow method
    """
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor],
                 initial_points: torch.Tensor,
                 optimizer: Callable[[torch.Tensor], torch.optim.Optimizer]):
        """
        Args:
            model (Callable[[torch.Tensor], torch.Tensor]): Function that maps
                a `torch.Tensor` of shape (N, D_in) to a tensor either of
                shape (N) and with values in [0,1] or of shape (N, D_out) with
                values in [0, 1] such that the last axis sums to 1.

            initial_points (torch.Tensor): `torch.Tensor` of shape (N, D_in)

            optimizer (Callable[[torch.Tensor], torch.optim.Optimizer]):
                Function returning an optimizer for the params given as an
                argument.
        """
        self.sample_points = initial_points
        self.sample_points.requires_grad = True

        output = model(self.sample_points)
        output_shape = output.size()

        if not len(output_shape) in [1, 2]:
            raise RuntimeError('Output of model has wrong size!')
        # Convert `model` to `self.model` with squared generalized distance
        # to the decision boundary as output.
        new_model = self._convert_to_distance_to_boundary(model, output_shape)
        self.model = lambda x: new_model(x)**2

        # Check if self.model has the right output shape
        assert len(self.model(self.sample_points).size()) == 1, \
            f'Output shape is {self.model(self.sample_points).size()}'

        self.optimizer = optimizer([self.sample_points])

    def step(self, number_of_steps=1):
        """Performs the indicated number of steps towards the decision boundary
        """
        for _ in range(number_of_steps):
            self.optimizer.zero_grad()
            loss = torch.sum(self.model(self.sample_points))
            loss.backward()
            self.optimizer.step()

    def get_decision_boundary(self) -> torch.Tensor:
        return self.sample_points

    def get_filtered_decision_boundary(self, delta=0.01) -> torch.Tensor:
        q_dist = self.model(self.sample_points)
        return self.sample_points[q_dist <= delta**2]


class QuasihyperbolicDecisionBoundaryCalculator(DecisionBoundaryCalculator):
    """
    Computes Decision Boundary by emanating quasihyperbolic geodesics
    """
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor],
                 initial_points: torch.Tensor,
                 initial_vectors: torch.Tensor,
                 integrator=None):
        """
        Args:
            model (Callable[[torch.Tensor], torch.Tensor]): Function that maps
                a `torch.Tensor` of shape (N, D_1, ..., D_k) to a tensor either of
                shape (N) and with values in [0,1] or of shape (N, num_classes) with
                values in [0, 1] such that the last axis sums to 1.

            initial_points (torch.Tensor): `torch.Tensor` of shape (N, D_1, ..., D_k)
                containing the starting points.

            initial_vectors(torch.Tensor): `torch.Tensor` of shape (N, D_1, ..., D_k)
                containing the starting tangent vectors (directions).
                Prefarably normalized.

            integrator: unused
        """
        self.input_shape = initial_points.shape # (N, D_1, ..., D_k)

        output = model(initial_points[:1])
        output_shape = output.shape # (1,) or (1, num_classes)

        if not len(output_shape) in [1, 2]:
            raise RuntimeError('Output of model has wrong size!')
        # Convert `model` to `self.model` with generalized distance
        # to the decision boundary as output.
        distance_fct = self._convert_to_distance_to_boundary(model, output_shape)

        # Check if `self.model` has the right output shape
        distance = distance_fct(initial_points)
        assert len(distance.size()) == 1, \
            f'Output shape is {distance.size()}'
        
        # Flatten self.points and self.vectors to (N, D_1*...* D_k)
        # and adjust input for distance function
        self.points = initial_points.reshape((self.input_shape[0],-1))
        # Normalize tangent vectors in quasihyperbolic metric
        self.vectors = torch.einsum('i,i...->i...',
                            distance,
                            initial_vectors
                        ).reshape((self.input_shape[0], -1))
        self.distance_fct = lambda x: distance_fct(x.reshape((-1,)+self.input_shape[1:]))

        self.integrator = integrator

    def step(self, number_of_steps=1):
        """Performs the indicated number of steps towards the decision boundary
        """

        # Calculate logarithmic gradient of generalized distance function at points
        def gradient(y):
            # self.points.grad.zero_()
            # loss = torch.sum(torch.log(self.model(self.points)))
            # loss.backward()
            
            # Code without adding an addition vector
            # y.requires_grad = True
            # loss = torch.sum(torch.log(self.model(y)))
            # loss.backward()
            # return y.grad.detach()
            
            # Code without adding an addition vector
            delta = torch.zeros_like(y, requires_grad=True)
            loss = torch.sum(torch.log(self.distance_fct(y + delta)))
            loss.backward()
            return delta.grad.detach()

        # quasi-hyperbolic geodesic equation
        def odes(t, x):
            y = x[0]
            dy = x[1]

            gradient_log_delta = gradient(y)

            # quasi-hyperbolic geodesic equation see markdown comment
            ddy = 2*torch.einsum('bi,bi,bj->bj', gradient_log_delta, dy, dy)\
                  - torch.einsum('bi,bj,bj->bi', gradient_log_delta, dy, dy)
            return torch.stack((dy, ddy))

        ode_initial_conditions = torch.stack((self.points, self.vectors))

        step_size = .5
        # t = torch.arange(0, 3e0, 2e-3)

        ode_solver_output = odeint(odes, ode_initial_conditions, 
                                   torch.tensor([0, number_of_steps * step_size],
                                   dtype=self.points.dtype, device=self.points.device),
                                   method='rk4', options=dict(step_size=step_size))
        self.points = ode_solver_output[1, 0]
        self.vectors = ode_solver_output[1, 1]

    def get_decision_boundary(self) -> torch.Tensor:
        """Return computed approximation of decision boundary.
        You have to call the step function before getting meaningful
        results.

        Returns:
            torch.Tensor: Tensor of shape (N, D_1, ..., D_k)
        """
        return self.points.reshape((-1,)+self.input_shape[1:])

    def get_filtered_decision_boundary(self, delta=0.01) -> torch.Tensor:
        """Return computed approximation of decision boundary filtered by
        distance function value <= delta.
        You have to call the step function before getting meaningful
        results.

        Args:
            delta (float, optional): Threshold. Defaults to 0.01.

        Returns:
            torch.Tensor: Tensor of shape (*, D_1, ..., D_k)
        """
        distance = self.distance_fct(self.points)
        return self.points[distance <= delta].reshape((-1,)+self.input_shape[1:])