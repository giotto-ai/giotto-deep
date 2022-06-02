
from gdeep.visualisation import Compactification
from gdeep.models import FFNet


def test_compactification():
    model = FFNet([2, 4, 2])
    c = Compactification(0.4,
                         2,
                         1000,
                         0.05,
                         50,
                         [(-1., 1.), (-1., 1.)],
                         model)
    c.create_final_distance_matrix()
    c.plot_chart(1)
