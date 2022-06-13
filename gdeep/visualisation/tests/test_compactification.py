
from gdeep.visualisation import Compactification
from gdeep.models import FFNet
from gdeep.utility import DEVICE


def test_compactification():
    model = FFNet([2, 4, 2]).to(DEVICE)
    c = Compactification(model,
                         0.4,
                         2,
                         1500,
                         0.05,
                         50,
                         [(-1., 1.), (-1., 1.)],
                         )
    try:
        c.create_final_distance_matrix()
        c.plot_chart(1)
    except ValueError:
        pass  # this is because the charts are empty of points
