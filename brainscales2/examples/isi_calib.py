import matplotlib.pyplot as plt
import numpy as np
import pynn_brainscales.brainscales2 as pynn

initial_values = {"threshold_v_threshold": 400,
                  "leak_v_leak": 1022,
                  "leak_i_bias": 950,
                  "reset_v_reset": 400,
                  "reset_i_bias": 950,
                  "threshold_enable": True,
                  "membrane_capacitance_capacitance": 32}


def get_isi(tau_ref: int):
    pynn.setup()
    initial_values.update({"refractory_period_refractory_time": tau_ref})
    pop = pynn.Population(1, pynn.cells.HXNeuron,
                          initial_values=initial_values)
    pop.record("spikes")
    pynn.run(0.2)

    spikes = pop.get_data("spikes").segments[0].spiketrains[0]
    if len(spikes) == 0:
        return 0
    isi = np.zeros(len(spikes) - 1)
    for i in range(len(spikes) - 1):
        isi[i] = spikes[i + 1] - spikes[i]

    pynn.end()

    return np.mean(isi)


# pylint: disable=too-many-locals
def calibrate_isi(target_isi: float):
    log = pynn.logger.get("isi_calib.calibrate_isi")
    best_isi_diff = target_isi
    best_tau_ref = 0
    best_isi = 0
    best_isi_std = 0
    refrac_periods = np.zeros(26)
    isis = np.zeros(26)
    isis_std = np.zeros(26)
    for i, tau_ref in enumerate(range(5, 256, 10)):
        sub_isi = np.zeros(5)
        for j, _ in enumerate(sub_isi):
            subisi = get_isi(tau_ref)
            if subisi == 0:
                subisi = get_isi(tau_ref)
            sub_isi[j] = subisi
        isi = np.mean(sub_isi)
        isi_std = np.std(sub_isi)
        refrac_periods[i] = tau_ref
        isis[i] = isi
        isis_std[i] = isi_std
        log.INFO("tau_ref: ", tau_ref, ",  ", "isi: ", isi)
        if abs(target_isi - isi) < best_isi_diff:
            best_isi_diff = abs(target_isi - isi)
            best_tau_ref = tau_ref
            best_isi = isi
            best_isi_std = isi_std
            log.INFO("This is the best run so far.")

    plt.figure()
    plt.xlabel("Refractory Period [LSB]")
    plt.ylabel("ISI [ms]")
    plt.errorbar(refrac_periods, isis, yerr=isis_std, color="blue",
                 linestyle="None", marker=".")
    plt.errorbar(best_tau_ref, best_isi, yerr=best_isi_std, color="red",
                 marker=".")
    plt.savefig("plot_isi_calib.pdf")
    plt.close()

    return best_isi, best_tau_ref


if __name__ == "__main__":
    pynn.logger.default_config(level=pynn.logger.LogLevel.INFO)
    result_isi, result_tau_ref = calibrate_isi(0.01)
    main_log = pynn.logger.get("isi_calib")
    main_log.INFO("Best result:")
    main_log.INFO("isi: ", result_isi, ", ", "tau_ref: ", result_tau_ref)
