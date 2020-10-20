from src.pv_array import PVArray
import os
import matplotlib.pyplot as plt

PV_PARAMS_PATH = os.path.join("parameters", "pvarray_01.json")

if __name__ == "__main__":
    pvarray = PVArray.from_json(PV_PARAMS_PATH)
    g = [1000, 1000, 1000, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900]
    t = [
        25,
    ] * len(g)

    true = pvarray.get_true_mpp(g, t)
    po = pvarray.get_po_mpp(g, t, v0=26, v_step=0.8)

    plt.plot(true.power, "o-", label="True P")
    plt.plot(po.power, "o-", label="PO P")
    plt.legend()
    plt.show()

    plt.plot(true.voltage, "o", label="True V")
    plt.plot(po.voltage, "o-", label="PO V")
    plt.legend()
    plt.show()